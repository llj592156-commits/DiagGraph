"""Runner plumbing tests; business acceptance uses real user-run extraction."""

import io
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lightrag.evaluation import printer_graph_smoke as smoke
from lightrag.evaluation.printer_graph_samples import query_slots

pytestmark = pytest.mark.offline


def sample(key="one", domain="network", models=None):
    return {
        "id": key,
        "doc_id": "doc-" + key,
        "document_name": "L3266_L5298.pdf",
        "parent_id": "p-" + key,
        "child_ids": ["c-" + key],
        "pages": [1],
        "source_spans": [{"block_id": "b", "start": 0, "end": 20}],
        "projection": {"kind": "parent_substring"},
        "domain": domain,
        "scope": {"applicable_models": models or ["L3266", "L5298"]},
        "heading_path": ["Guide"],
        "input_text": "E-12 Network issue. Check IP.",
    }


def extraction():
    return {
        "entities": [
            {
                "name": "E-12",
                "type": "FaultCode",
                "description": "Network issue",
                "evidence_quotes": ["E-12"],
            },
            {
                "name": "Check IP",
                "type": "Procedure",
                "description": "Check IP",
                "evidence_quotes": ["Check IP"],
            },
        ],
        "relationships": [
            {
                "source": "E-12",
                "target": "Check IP",
                "relation": "HANDLED_BY",
                "description": "Check IP",
                "evidence_quotes": ["Check IP"],
            }
        ],
    }


def response(value=None):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(value or extraction())},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def settings():
    return {
        "base_url": "https://example.test/v1",
        "model": "extract-test",
        "api_key": "secret",
        "extra_body": {"enable_thinking": False},
        "timeout": 30,
        "max_tokens": 4096,
        "temperature": 0,
    }


@pytest.mark.parametrize(
    "damage", ["quote", "endpoint", "type", "truncated", "json", "code", "duplicate"]
)
def test_unusable_outputs_are_retained_for_review_not_silently_repaired(damage):
    value = extraction()
    if damage == "quote":
        value["relationships"][0]["evidence_quotes"] = ["Not in the source"]
    elif damage == "endpoint":
        value["relationships"][0]["target"] = "Missing"
    elif damage == "type":
        value["entities"][0]["type"] = "UnknownType"
    elif damage == "code":
        value["entities"][0]["name"] = "E-99"
    elif damage == "duplicate":
        value["entities"].append(deepcopy(value["entities"][0]))
    payload = response(value)
    if damage == "truncated":
        payload["choices"][0]["finish_reason"] = "length"
    elif damage == "json":
        payload["choices"][0]["message"]["content"] = "```json\n{}\n```"
    with pytest.raises(ValueError):
        smoke.validate_extraction(payload, sample())


def test_domain_identity_and_source_filters_use_the_same_extracted_facts():
    first, second = sample(), sample("two", "device_display_errors", ["L4263"])
    extracted = {s["id"]: extraction() for s in (first, second)}
    before = deepcopy(extracted)
    raw = smoke.build_graph([first, second], extracted, False)
    scoped = smoke.build_graph([first, second], extracted, True)
    query = {"id": "Q", "query": "L3266 网络连接报告出现 E-12"}
    assert len(smoke.query_graph(raw, query, False)["evidence"]) == 2
    assert len(smoke.query_graph(scoped, query, True)["evidence"]) == 1
    ambiguous = smoke.query_graph(scoped, {"id": "Q", "query": "E-12"}, True)
    assert ambiguous["status"] == "needs_context" and ambiguous["evidence"] == []
    assert len(ambiguous["candidate_facts"]) == 2
    assert extracted == before
    assert all(d["source"]["child_ids"] for _, _, d in scoped.edges(data=True))


def test_slots_are_from_question_not_reference_labels():
    assert query_slots("L4263 显示屏 W-11")["domain"] == "device_display_errors"
    assert query_slots("L5298 网络连接报告 E-12")["codes"] == ["E-12"]
    assert query_slots("出现 E-12")["domain"] is None


def test_request_uses_extract_configuration_without_other_models(monkeypatch):
    captured = []

    def open_request(req, timeout):
        captured.append(json.loads(req.data))
        return io.StringIO(json.dumps(response()))

    monkeypatch.setattr(
        smoke.request, "build_opener", lambda *args: SimpleNamespace(open=open_request)
    )
    result = smoke.call_model(settings(), smoke.messages(sample()))
    assert result["choices"]
    assert captured[0]["model"] == "extract-test"
    assert captured[0]["enable_thinking"] is False
    assert captured[0]["max_tokens"] == 4096
    assert captured[0]["stream"] is False
    assert "secret" not in json.dumps(captured)


def test_preflight_and_resume_do_not_repeat_successful_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "config", lambda *args: settings())
    monkeypatch.setattr(smoke, "load_samples", lambda *args: ([sample()], {}))
    monkeypatch.setattr(
        smoke,
        "load_local_tokenizer",
        lambda *args: SimpleNamespace(encode=lambda text: list(text)),
    )
    calls = []
    monkeypatch.setattr(
        smoke, "call_model", lambda *args: calls.append(args) or response()
    )
    args = ["--output", str(tmp_path / "output")]
    assert smoke.main(args) == 0 and calls == []
    assert "secret" not in (tmp_path / "output/experiment.json").read_text(
        encoding="utf-8"
    )
    assert smoke.main(args + ["--run", "--resume"]) == 0
    assert len(calls) == 1
    assert smoke.main(args + ["--run", "--resume"]) == 0
    assert len(calls) == 1
    assert (tmp_path / "output/raw/one.json").exists()


def test_config_resolves_extract_role_and_never_exports_key(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke.os, "environ", {})
    envfile = tmp_path / "test.env"
    envfile.write_text(
        'LLM_BINDING=openai\nLLM_MODEL=base\nLLM_BINDING_API_KEY=base-secret\nEXTRACT_LLM_MODEL=role\nEXTRACT_LLM_BINDING_API_KEY=role-secret\nEXTRACT_OPENAI_LLM_EXTRA_BODY={"enable_thinking":false}\n',
        encoding="utf-8",
    )
    result = smoke.config(envfile, 4096)
    assert result["model"] == "role" and result["api_key"] == "role-secret"
    assert "secret" not in json.dumps(smoke.public_config(result))


def test_query_oracles_never_enter_model_messages():
    encoded = json.dumps(smoke.messages(sample()), ensure_ascii=False)
    for query in smoke.QUERIES:
        assert query["query"] not in encoded


def test_sample_projection_keeps_inline_note_with_exact_source_attribution():
    from lightrag.evaluation.printer_graph_samples import table_row
    from tests.document_structure.test_children import parent

    text = '<table id="tb" format="json">[["Code","Solution"],["W-11","Resume *1"]]</table>\n*1 Resuming is not repair.\n\n## Next\nOther text'
    p = parent(text=text)
    p["pages"] = [1]
    idx = SimpleNamespace(
        children={
            "c": {
                "child_id": "c",
                "parent_id": p["parent_id"],
                "source_spans": p["source_spans"],
            }
        }
    )
    result = table_row(idx, p, "W-11", "S")
    assert "*1 Resuming is not repair." in result["input_text"]
    assert "Other text" not in result["input_text"]
    note = result["inline_notes"][0]
    assert (
        p["content"][note["parent_span"]["start"] : note["parent_span"]["end"]]
        == note["content"]
    )
    assert note["source_spans"][0]["start"] == 10 + note["parent_span"]["start"]
