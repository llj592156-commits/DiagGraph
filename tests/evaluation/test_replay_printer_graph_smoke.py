"""Saved model output review preserves provenance and rejects unsupported fixes."""

import json
from copy import deepcopy

import pytest

from lightrag.evaluation import replay_printer_graph_smoke as replay
from lightrag.evaluation.compare_parent_child_retrieval import digest

from .test_printer_graph_smoke import extraction, response, sample

pytestmark = pytest.mark.offline


def test_bad_relation_does_not_discard_other_real_records():
    value = extraction()
    value["relationships"].append(
        {**value["relationships"][0], "evidence_quotes": ["Not in source"]}
    )
    result = replay.review_response(response(value), sample())
    assert not result["strict_passed"]
    assert len(result["accepted"]["entities"]) == 2
    assert len(result["accepted"]["relationships"]) == 1
    assert result["rejected"][0]["path"] == "relationships[1]"


def test_only_verified_outer_quote_format_is_normalized_and_audited():
    value = extraction()
    value["entities"][1]["evidence_quotes"] = ['"Check IP"']
    original = deepcopy(value)
    result = replay.review_response(response(value), sample())
    assert result["accepted"]["entities"][1]["evidence_quotes"] == ["Check IP"]
    assert result["format_normalizations"][0]["original"] == '"Check IP"'
    assert result["format_normalizations"][0]["rule"] == "one_matching_outer_quote_pair"
    assert value == original


@pytest.mark.parametrize(
    "quote",
    ["E-12 ... Check IP", '"E-12 ... Check IP"', "E-12 Check IP", "'Not in source'"],
)
def test_noncontiguous_or_invented_quotes_are_never_reconstructed(quote):
    value = extraction()
    value["relationships"][0]["evidence_quotes"] = [quote]
    result = replay.review_response(response(value), sample())
    assert result["accepted"]["relationships"] == []
    assert result["rejected"] and result["format_normalizations"] == []


def test_unknown_relation_stays_unknown_instead_of_being_aliased():
    value = extraction()
    value["relationships"][0]["relation"] = "DEPENDS_ON"
    result = replay.review_response(response(value), sample())
    assert result["accepted"]["relationships"] == []
    assert result["rejected"][0]["original_record"]["relation"] == "DEPENDS_ON"


def test_rejected_entity_cannot_leave_an_orphan_relation():
    value = extraction()
    value["entities"][1]["evidence_quotes"] = ["Not in source"]
    result = replay.review_response(response(value), sample())
    assert len(result["accepted"]["entities"]) == 1
    assert result["accepted"]["relationships"] == []
    assert len(result["rejected"]) == 2


def test_truncated_output_is_not_salvaged_as_a_complete_sample():
    payload = response()
    payload["choices"][0]["finish_reason"] = "length"
    result = replay.review_response(payload, sample())
    assert result["sample_error"]
    assert result["accepted"] == {"entities": [], "relationships": []}


def saved_run(path):
    path.mkdir()
    plan = {
        "version": "printer-real-graph-smoke-v1",
        "samples": [sample()],
        "queries": [],
        "provider": {"model": "test"},
    }
    plan["plan_hash"] = digest(plan)
    (path / "experiment.json").write_text(json.dumps(plan), encoding="utf-8")
    (path / "samples.json").write_text(json.dumps(plan["samples"]), encoding="utf-8")
    (path / "raw").mkdir()
    (path / "raw/one.json").write_text(
        json.dumps(
            {"plan_hash": plan["plan_hash"], "sample_id": "one", "response": response()}
        ),
        encoding="utf-8",
    )
    return plan


def test_replay_never_calls_models_or_modifies_source(monkeypatch, tmp_path):
    from lightrag.evaluation import printer_graph_smoke

    source, output = tmp_path / "source", tmp_path / "review"
    saved_run(source)
    before = {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        printer_graph_smoke,
        "call_model",
        lambda *args: pytest.fail("No remote calls in review"),
    )
    args = ["--source-directory", str(source), "--output", str(output)]
    assert replay.main(args + ["--preflight"]) == 0
    assert not output.exists()
    assert replay.main(args) == 0
    assert (output / "report.md").exists()
    assert {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    } == before
    with pytest.raises(ValueError, match="exists"):
        replay.main(args)


@pytest.mark.parametrize("damage", ["plan", "samples", "raw"])
def test_foreign_or_mutated_source_artifacts_are_refused(tmp_path, damage):
    source = tmp_path / "source"
    plan = saved_run(source)
    if damage == "plan":
        plan["provider"]["model"] = "different"
        (source / "experiment.json").write_text(json.dumps(plan), encoding="utf-8")
    elif damage == "samples":
        (source / "samples.json").write_text("[]", encoding="utf-8")
    else:
        raw = json.loads((source / "raw/one.json").read_text(encoding="utf-8"))
        raw["sample_id"] = "different"
        (source / "raw/one.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        replay.load_saved(source)
