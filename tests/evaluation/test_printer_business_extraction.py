"""Mock transport checks; real extraction acceptance is performed by the user."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lightrag.evaluation import printer_business_extraction as runner
from tests.extraction.test_printer_contract import response, sample

pytestmark = pytest.mark.offline


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    source = {
        **sample(),
        "id": "S01",
        "doc_id": "doc-B",
        "parent_id": "p-B",
        "child_ids": ["c-B"],
        "pages": [183],
        "scope": {"applicable_models": ["L3266", "L5298"], "status": "filename"},
        "source_spans": [{"block_id": "block-B", "start": 0, "end": 30}],
        "projection": {"kind": "parent_substring"},
    }
    monkeypatch.setattr(
        runner, "load_samples", lambda *_: ([source], {"B": "sha256:source"})
    )
    monkeypatch.setattr(
        runner,
        "load_local_tokenizer",
        lambda *_: SimpleNamespace(encode=lambda t: list(t)),
    )
    monkeypatch.setattr(
        runner,
        "config",
        lambda *_: {
            "base_url": "https://example.test/v1",
            "model": "extract-model",
            "api_key": "test-secret",
            "max_tokens": 6144,
            "extra_body": {},
            "temperature": 0,
            "timeout": 300,
        },
    )
    calls = []
    monkeypatch.setattr(
        runner, "call_model", lambda *args: calls.append(args) or response()
    )
    output = tmp_path / "run"
    return source, output, ["--output", str(output)], calls


def test_preflight_is_local_and_freezes_no_api_key(prepared):
    _, output, args, calls = prepared
    assert runner.main(args) == 0
    assert calls == []
    assert not (output / "raw").exists()
    assert not (output / "embedding.lock").exists()
    plan = (output / "experiment.json").read_text(encoding="utf-8")
    assert "test-secret" not in plan
    assert json.loads(plan)["contract_version"] == runner.printer.VERSION
    assert (output / "prompts.json").exists()


def test_run_and_resume_preserve_evidence_scope_and_do_not_repeat_calls(prepared):
    source, output, args, calls = prepared
    before = deepcopy(source)
    assert runner.main(args) == 0
    assert runner.main(args + ["--run", "--resume"]) == 0
    raw = (output / "raw/S01.json").read_bytes()
    assert runner.main(args + ["--run", "--resume"]) == 0
    assert len(calls) == 1
    assert (output / "raw/S01.json").read_bytes() == raw
    rows = json.loads((output / "evidence_records.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    assert all(row["source_metadata"]["scope"] == source["scope"] for row in rows)
    assert all(row["source_metadata"]["child_ids"] == ["c-B"] for row in rows)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["new_requests_this_run"] == 0
    assert summary["status"] == "needs_semantic_review"
    assert summary["embedding_requests"] == summary["answer_generation_requests"] == 0
    assert source == before
    assert not list(output.glob("graph_*.json"))


def test_changed_contract_refuses_resume_before_calling_model(prepared, monkeypatch):
    _, _, args, calls = prepared
    assert runner.main(args) == 0
    monkeypatch.setattr(runner.printer, "VERSION", "changed-contract")
    with pytest.raises(ValueError, match="Frozen plan"):
        runner.main(args + ["--run", "--resume"])
    assert calls == []


def test_foreign_output_is_never_overwritten(prepared):
    _, output, args, calls = prepared
    output.mkdir()
    marker = output / "important.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="foreign output"):
        runner.main(args + ["--resume"])
    assert marker.read_text(encoding="utf-8") == "keep"
    assert calls == []


def test_invalid_records_are_quarantined_and_saved_response_is_reused(
    prepared, monkeypatch
):
    _, output, args, calls = prepared
    value = response()
    body = json.loads(value["choices"][0]["message"]["content"])
    body["relationships"][0]["qualifiers"][0]["evidence_quotes"] = ["省略……内容"]
    value["choices"][0]["message"]["content"] = json.dumps(body)
    monkeypatch.setattr(runner, "call_model", lambda *a: calls.append(a) or value)
    assert runner.main(args + ["--run"]) == 2
    assert runner.main(args + ["--run", "--resume"]) == 2
    assert len(calls) == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_entities"] == 3
    assert summary["accepted_relationships"] == 1
    assert summary["rejected_records"] == 1


def test_tampered_cache_identity_is_rejected(prepared):
    _, output, args, calls = prepared
    assert runner.main(args + ["--run"]) == 0
    path = output / "raw/S01.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sample_id"] = "other-sample"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Cached response"):
        runner.main(args + ["--resume", "--run"])
    assert len(calls) == 1


def test_empty_model_result_is_flagged_instead_of_claiming_success(
    prepared, monkeypatch
):
    _, output, args, _ = prepared
    monkeypatch.setattr(
        runner, "call_model", lambda *_: response({"entities": [], "relationships": []})
    )
    assert runner.main(args + ["--run"]) == 2
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["empty_samples"] == 1


def test_interruption_resumes_only_missing_response(prepared, monkeypatch):
    source, output, args, calls = prepared
    second = {**source, "id": "S02"}
    monkeypatch.setattr(runner, "load_samples", lambda *_: ([source, second], {}))

    def flaky(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("Connection failed")
        return response()

    monkeypatch.setattr(runner, "call_model", flaky)
    with pytest.raises(RuntimeError, match="Connection failed"):
        runner.main(args + ["--run"])
    assert (output / "raw/S01.json").exists()
    assert not (output / "raw/S02.json").exists()
    assert not (output / "embedding.lock").exists()
    assert runner.main(args + ["--resume", "--run"]) == 0
    assert len(calls) == 3
