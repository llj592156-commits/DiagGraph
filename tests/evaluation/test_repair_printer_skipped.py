"""Bounded skipped-source repair preserves anchors and replays saved responses."""

import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from lightrag.evaluation import repair_printer_skipped as runner
from lightrag.extraction.runtime import snapshot_config
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline


class Characters:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)


def response(text=None, truncated=False):
    if text is None:
        text = json.dumps(
            {
                "entities": [
                    {
                        "name": "E-12",
                        "type": "FaultCode",
                        "description": "故障代码",
                        "evidence_quotes": ["E-12"],
                    },
                    {
                        "name": "故障",
                        "type": "Symptom",
                        "description": "故障现象",
                        "evidence_quotes": ["故障"],
                    },
                ],
                "relationships": [
                    {
                        "source": "E-12",
                        "target": "故障",
                        "relation": "INDICATES",
                        "description": "代码表示故障",
                        "evidence_quotes": ["E-12 表示故障"],
                        "qualifiers": [],
                    }
                ],
            },
            ensure_ascii=False,
        )
    return {
        "choices": [
            {
                "finish_reason": "length" if truncated else "stop",
                "message": {"content": text},
            }
        ]
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    storage = tmp_path / "storage" / "test_repair"
    storage.mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    key = "doc-original-pc-child-skipped"
    chunk = {
        "content": "E-12 表示故障，请联系服务机构。",
        "full_doc_id": "doc-original",
        "file_path": "L5298_UG.pdf",
        "tokens": 20,
        "chunk_order_index": 0,
        "extraction_config": snapshot_config("printer_v2"),
        "parent_child": {
            "doc_id": "doc-original",
            "parent_id": "parent-1",
            "heading_path": ["显示屏错误代码"],
            "feature_domain": "device_display_errors",
        },
        "business_extraction": {"nodes": {}, "facts": [], "review": {"skipped": True}},
        "business_extraction_attempt": {"raw_response": "broken"},
        "llm_cache_list": ["default:extract:bad"],
    }
    files = {
        "text_chunks": {
            key: chunk,
            "untouched": {"content": "unchanged", "full_doc_id": "doc-original"},
        },
        "llm_response_cache": {
            "default:extract:bad": {
                "chunk_id": key,
                "cache_type": "extract",
                "original_prompt": "frozen prompt",
                "return": "broken",
            }
        },
        "doc_status": {
            "doc-original": {
                "status": "processed",
                "chunks_list": [key, "untouched"],
                "chunks_count": 2,
                "content_summary": "fixture",
                "content_length": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "file_path": "L5298_UG.pdf",
            }
        },
        "full_docs": {
            "doc-original": {
                "content": "original",
                "parent_child": {
                    "parents": {"parent-1": {"content": chunk["content"]}}
                },
            }
        },
        "full_entities": {"doc-original": {"entity_names": ["old-entity"], "count": 1}},
        "full_relations": {
            "doc-original": {"relation_pairs": [["old-left", "old-right"]], "count": 1}
        },
    }
    for namespace, rows in files.items():
        runner.write_json(storage / f"kv_store_{namespace}.json", rows)
    skipped = tmp_path / "skipped.json"
    runner.write_json(
        skipped, [{"chunk_id": key, "doc_id": "doc-original", "parent_id": "parent-1"}]
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"WORKING_DIR={tmp_path / 'storage'}\nWORKSPACE=test_repair\n", encoding="utf-8"
    )
    # Isolate the user's real workspace/provider defaults in imported fixtures.
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("WORKSPACE", "test_repair")
    monkeypatch.setattr(runner, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(
        runner,
        "config",
        lambda *a: {
            "api_key": "offline",
            "model": "mock",
            "base_url": "https://mock.invalid",
            "timeout": 10,
            "max_tokens": 6144,
            "extra_body": {},
        },
    )
    monkeypatch.setattr(
        runner,
        "embedding_config",
        lambda *a: {
            "api_key": "offline",
            "dimension": 8,
            "token_limit": 3072,
            "model": "mock",
            "base_url": "https://mock.invalid",
        },
    )
    monkeypatch.setattr(runner, "require_stopped", lambda *a: None)
    monkeypatch.setattr(
        runner, "load_local_tokenizer", lambda *a: Tokenizer("chars", Characters())
    )
    args = argparse.Namespace(
        env_file=env_file,
        skipped_file=skipped,
        output=output,
        max_output_tokens=6144,
        tokenizer=Path("unused"),
        resume=False,
    )
    result = runner.prepare(args)
    runner.write_json(output / "plan.json", result[0])
    return args, result, storage, key, files


@pytest.mark.parametrize("failure", ["good", "invalid", "truncated", "empty"])
def test_only_targeted_calls_and_native_repair_with_mock_providers(
    prepared, monkeypatch, failure
):
    args, ready, storage, key, originals = prepared
    calls = []

    async def extract(settings, messages, response_format, **kwargs):
        calls.append(messages)
        if failure == "invalid":
            return response("broken")
        if failure == "empty":
            return response('{"entities": [], "relationships": []}')
        return response(truncated=failure == "truncated")

    monkeypatch.setattr(runner, "request_extraction", extract)
    monkeypatch.setattr(
        runner,
        "embed_batch",
        lambda settings, texts: (np.ones((len(texts), 8), dtype=np.float32), {}),
    )
    code = asyncio.run(runner.run(args, ready))
    assert len(calls) == 1
    current = runner.read_json(storage / "kv_store_text_chunks.json")
    assert runner.source_hash(current[key]) == runner.source_hash(
        originals["text_chunks"][key]
    )
    assert current["untouched"]["content"] == "unchanged"
    assert (args.output / "workspace_before.zip").exists()
    anchors = runner.read_json(storage / "kv_store_full_entities.json")["doc-original"]
    assert "old-entity" in anchors["entity_names"]
    assert ["old-left", "old-right"] in runner.read_json(
        storage / "kv_store_full_relations.json"
    )["doc-original"]["relation_pairs"]
    cache = runner.read_json(storage / "kv_store_llm_response_cache.json")[
        "default:extract:bad"
    ]
    if failure in {"invalid", "truncated"}:
        assert code == 2
        assert current[key]["business_extraction"]["review"]["skipped"]
        assert cache["return"] == "broken"
    else:
        assert code == 0
        assert runner.completed(current[key], ready[0]["plan_hash"])
        assert not current[key]["business_extraction"]["review"].get("skipped")
        assert cache["return"] == calls_response(failure)
        before = deepcopy(current[key]["business_extraction"])
        # A second invocation is a no-op even after reloading file-backed stores.
        new_ready = runner.prepare(args)
        assert asyncio.run(runner.run(args, new_ready)) == 0
        assert len(calls) == 1
        assert (
            runner.read_json(storage / "kv_store_text_chunks.json")[key][
                "business_extraction"
            ]
            == before
        )


def calls_response(failure):
    return response(
        '{"entities": [], "relationships": []}' if failure == "empty" else None
    )["choices"][0]["message"]["content"]


@pytest.mark.parametrize("change", ["source", "shared_cache", "accepted", "too_many"])
def test_preflight_refuses_wrong_source_and_unbounded_repair(prepared, change):
    args, _, storage, key, _ = prepared
    chunks = runner.read_json(storage / "kv_store_text_chunks.json")
    if change == "source":
        chunks[key]["content"] += " changed"
    elif change == "shared_cache":
        chunks["untouched"]["llm_cache_list"] = chunks[key]["llm_cache_list"]
    elif change == "accepted":
        chunks[key]["business_extraction"]["nodes"] = {"existing": {}}
    else:
        runner.write_json(args.skipped_file, runner.read_json(args.skipped_file) * 11)
    runner.write_json(storage / "kv_store_text_chunks.json", chunks)
    with pytest.raises(ValueError):
        runner.prepare(args)


def test_resume_reuses_valid_response_after_graph_failure(prepared, monkeypatch):
    args, ready, storage, key, _ = prepared
    calls = []

    async def extract(*a, **kw):
        calls.append(1)
        return response()

    original_merge = runner.merge_nodes_and_edges

    async def fail_after_anchors(**kwargs):
        anchors = runner.read_json(storage / "kv_store_full_entities.json")[
            "doc-original"
        ]
        assert "old-entity" in anchors["entity_names"]
        assert len(anchors["entity_names"]) > 1
        assert kwargs["full_entities_storage"] is None
        raise RuntimeError("mock graph failure")

    monkeypatch.setattr(runner, "request_extraction", extract)
    monkeypatch.setattr(
        runner,
        "embed_batch",
        lambda settings, texts: (np.ones((len(texts), 8), dtype=np.float32), {}),
    )
    monkeypatch.setattr(runner, "merge_nodes_and_edges", fail_after_anchors)
    with pytest.raises(RuntimeError, match="mock graph failure"):
        asyncio.run(runner.run(args, ready))
    assert runner.read_json(storage / "kv_store_text_chunks.json")[key][
        "business_extraction"
    ]["review"]["skipped"]
    args.resume = True
    monkeypatch.setattr(runner, "merge_nodes_and_edges", original_merge)
    assert asyncio.run(runner.run(args, runner.prepare(args))) == 0
    assert len(calls) == 1


def test_live_server_is_refused_before_api_calls(monkeypatch):
    class Connection:
        def close(self):
            pass

    monkeypatch.setattr(
        runner.socket, "create_connection", lambda *a, **kw: Connection()
    )
    with pytest.raises(ValueError, match="Stop the LightRAG"):
        runner.require_stopped({"PORT": "9621"})


def test_pending_cache_resume_does_not_repeat_graph_merge_or_model(
    prepared, monkeypatch
):
    args, ready, storage, key, _ = prepared
    calls, embeds = [], []

    async def extract(*a, **kw):
        calls.append(1)
        return response()

    def embed(settings, texts):
        embeds.append(len(texts))
        return np.ones((len(texts), 8), dtype=np.float32), {}

    original_flush = runner.LightRAG._flush_storages
    failed = False

    async def fail_cache_flush(self, stores):
        nonlocal failed
        if stores == [self.llm_response_cache] and not failed:
            failed = True
            raise RuntimeError("mock cache publication failure")
        return await original_flush(self, stores)

    monkeypatch.setattr(runner, "request_extraction", extract)
    monkeypatch.setattr(runner, "embed_batch", embed)
    monkeypatch.setattr(runner.LightRAG, "_flush_storages", fail_cache_flush)
    with pytest.raises(RuntimeError, match="mock cache publication failure"):
        asyncio.run(runner.run(args, ready))
    assert (
        runner.read_json(storage / "kv_store_text_chunks.json")[key][
            "business_extraction_repair"
        ]["phase"]
        == "pending_cache"
    )
    embed_calls_before = list(embeds)
    args.resume = True
    assert asyncio.run(runner.run(args, runner.prepare(args))) == 0
    assert calls == [1]
    assert embeds == embed_calls_before


@pytest.mark.parametrize("malformed", [False, True])
def test_plain_json_retry_changes_transport_only_and_keeps_validation(
    prepared, monkeypatch, malformed
):
    args, ready, _, key, _ = prepared
    args.resume = True
    args.plain_json_retry = True
    messages = runner.extraction_messages(key, ready[1][key])

    async def extract(settings, actual_messages, response_format, **kwargs):
        assert response_format is None
        assert actual_messages == messages
        return response("broken" if malformed else None)

    monkeypatch.setattr(runner, "request_extraction", extract)
    counters = {"extraction_requests": 0, "extraction_api_attempts": 0}
    valid, failures = asyncio.run(
        runner.extract_targets(args, ready[0], ready[1], ready[3], counters)
    )
    assert (key in valid) is not malformed
    assert (key in failures) is malformed
    assert runner.read_json(args.output / "plan.json") == ready[0]
    requests = list(args.output.glob("request_*.json"))
    assert len(requests) == 1
    assert runner.read_json(requests[0])["response_format"] is None


def test_mixed_quote_repair_only_inserts_escapes_and_keeps_original_response():
    expected = {"description": '用于将“保存到 PC"设置为是', "other": "正常值"}
    correct_text = json.dumps(expected, ensure_ascii=False)
    broken = correct_text.replace('PC\\"设置', 'PC"设置')
    original = response(broken)
    saved = deepcopy(original)
    derived, edits = runner.repair_mixed_quotes(original)
    assert original == saved
    assert json.loads(derived["choices"][0]["message"]["content"]) == expected
    assert len(edits) == 1
    assert edits[0]["insert"] == "\\"


@pytest.mark.parametrize(
    "saved",
    [
        response("broken"),
        response('{"entities":'),
        response(truncated=True),
        response(),
    ],
)
def test_saved_quote_repair_refuses_other_errors_without_inventing_fields(saved):
    with pytest.raises(ValueError):
        runner.repair_mixed_quotes(saved)


def test_saved_quote_repair_calls_no_model_and_persists_audited_derivative(
    prepared, monkeypatch
):
    args, ready, storage, key, _ = prepared
    args.resume = True
    args.repair_saved_quotes = True
    expected = json.loads(response()["choices"][0]["message"]["content"])
    expected["entities"][0]["description"] = '用于将“保存到 PC"设置为是'
    broken = json.dumps(expected, ensure_ascii=False).replace('PC\\"设置', 'PC"设置')
    saved = {
        "response": response(broken),
        "attempt": 1,
        "timestamp": 1,
        "validation_failed": True,
    }
    path = args.output / f"response_{key}.json"
    runner.write_json(path, saved)

    async def refused(*a, **kw):
        raise AssertionError("Saved-response recovery must never call extraction API")

    monkeypatch.setattr(runner, "request_extraction", refused)
    monkeypatch.setattr(
        runner,
        "embed_batch",
        lambda settings, texts: (np.ones((len(texts), 8), dtype=np.float32), {}),
    )
    assert asyncio.run(runner.run(args, ready)) == 0
    report = runner.read_json(args.output / "summary.json")
    assert report["calls_this_run"]["extraction_requests"] == 0
    assert report["repaired_chunks"] == 1
    audit = runner.read_json(args.output / f"syntax_repair_{key}_attempt_1.json")
    assert audit["original_response"] == saved["response"]
    assert len(audit["changes"]) == 1
    chunk = runner.read_json(storage / "kv_store_text_chunks.json")[key]
    assert (
        chunk["business_extraction_repair"]["syntax_repair"]["inserted_backslashes"]
        == 1
    )


@pytest.mark.parametrize("missing", [False, True])
def test_saved_quote_repair_never_falls_back_to_model_on_unrecoverable_input(
    prepared, monkeypatch, missing
):
    args, ready, _, key, _ = prepared
    args.resume = True
    args.repair_saved_quotes = True
    if not missing:
        runner.write_json(
            args.output / f"response_{key}.json",
            {
                "response": response("broken"),
                "attempt": 1,
                "timestamp": 1,
                "validation_failed": True,
            },
        )

    async def refused(*a, **kw):
        raise AssertionError(
            "Unrecoverable saved output must not trigger paid extraction"
        )

    monkeypatch.setattr(runner, "request_extraction", refused)
    counters = {"extraction_requests": 0, "extraction_api_attempts": 0}
    valid, failed = asyncio.run(
        runner.extract_targets(args, ready[0], ready[1], ready[3], counters)
    )
    assert valid == {}
    assert key in failed
    assert counters["extraction_requests"] == 0
