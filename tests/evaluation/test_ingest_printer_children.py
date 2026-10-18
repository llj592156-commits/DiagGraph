"""Saved-child sampling feeds real native stores; providers are mocked."""

import argparse
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag.document_structure.children import load_local_tokenizer
from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.evaluation import ingest_printer_children as runner
from lightrag.extraction import printer
from lightrag.extraction import runtime as business_runtime
from lightrag.extraction.errors import (
    ExtractionRequestError,
    ExtractionRetriesExhausted,
)
from lightrag.extraction.runtime import snapshot_config

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[2]
BUNDLES = ROOT / "outputs/printer_parent_child/20261017_v3"


def index():
    if not BUNDLES.exists():
        pytest.skip("Local saved printer bundles unavailable")
    return ParentChildIndex.from_directory(BUNDLES / "B")


def test_selection_and_import_keep_complete_saved_child_and_source_mapping():
    saved = index()
    chosen = runner.select_children(saved, 4)
    tokenizer = load_local_tokenizer(
        ROOT / "tmp/parent_blocks_tools/cl100k_base.tiktoken"
    )
    chunks, bundle = runner.prepared_chunks(
        chosen, saved, "doc-small", "L3266_L5298.txt", tokenizer
    )
    assert len(chunks) == 4
    for child, chunk in zip(chosen, chunks, strict=True):
        assert child == saved.children[child["child_id"]]
        assert chunk["content"] == runner.child_evidence_content(child)
        assert chunk["parent_child"]["child_id"] == child["child_id"]
        assert chunk["parent_child"]["source_doc_id"] == child["doc_id"]
        assert chunk["parent_child"]["doc_id"] == "doc-small"
        assert chunk["parent_child"]["scope"] == child["scope"]
        parent = bundle["parents"][child["parent_id"]]
        assert parent["indexed_chunk_ids"]
        assert parent["content"] == saved.parents[child["parent_id"]]["content"]


@pytest.mark.parametrize(
    "resume_case",
    [
        "none",
        "over_limit",
        "invalid_json",
        "partial_invalid",
        "missing_section",
        "truncated",
        "parallel",
        "transport",
    ],
)
@pytest.mark.parametrize("full_ingestion", [False, True])
def test_real_native_ingestion_and_exports_with_mock_providers(
    tmp_path, monkeypatch, caplog, resume_case, full_ingestion
):
    saved = index()
    child_count = (
        2 if resume_case in {"partial_invalid", "parallel", "transport"} else 1
    )
    chosen = runner.select_children(saved, child_count)
    tokenizer = load_local_tokenizer(
        ROOT / "tmp/parent_blocks_tools/cl100k_base.tiktoken"
    )
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "100000")
    monkeypatch.setattr(runner, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "EXTRACTION_RETRY_DELAYS", (0, 0))

    def offline_socket(*a, **kw):
        raise OSError("Backend stopped")

    monkeypatch.setattr(runner.socket, "create_connection", offline_socket)

    model_calls = []
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = peak_active = 0

    def response(settings, messages, *, response_format):
        nonlocal active, peak_active
        assert response_format == {"type": "json_object"}
        model_calls.append(messages)
        if resume_case == "transport" and len(model_calls) <= 3:
            raise ExtractionRequestError(
                "Extraction connection/timeout failure", retryable=True
            )
        if resume_case == "parallel":
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                barrier.wait(timeout=5)
                if model_calls[0] is messages:
                    time.sleep(0.02)
            finally:
                with lock:
                    active -= 1
        assert messages[0]["content"].startswith("Extract an evidence-grounded")
        text = json.loads(messages[1]["content"])["source_text"]
        assert "E-12" in text
        if resume_case == "over_limit":
            entities = [
                {
                    "name": f"Object{i}",
                    "type": "Other",
                    "description": "Software fixture",
                    "evidence_quotes": ["E-12"],
                }
                for i in range(16)
            ]
        else:
            entities = [
                {
                    "name": "E-12",
                    "type": "FaultCode",
                    "description": "原文中的代码",
                    "evidence_quotes": ["E-12"],
                }
            ]
        return {
            "request_fingerprint": runner.fingerprint(messages[1]["content"]),
            "choices": [
                {
                    "finish_reason": "length" if resume_case == "truncated" else "stop",
                    "message": {
                        "content": (
                            '{"entities": [], "relationships": ["bad" "JSON"]}'
                            if resume_case in {"invalid_json", "partial_invalid"}
                            and len(model_calls) == 1
                            else json.dumps(
                                {"entities": entities}
                                if resume_case == "missing_section"
                                else {
                                    "entities": entities,
                                    "relationships": [],
                                }
                            )
                        )
                    },
                }
            ],
        }

    monkeypatch.setattr(runner, "call_model", response)
    monkeypatch.setattr(
        runner, "embed_batch", lambda settings, texts: (np.ones((len(texts), 8)), {})
    )
    plan = {
        "plan_hash": "test-plan",
        "working_dir": str(tmp_path / "existing-store"),
        "workspace": f"current-{tmp_path.name}",
        "stores": {k: default for k, (_, default) in runner.STORES.items()},
        "contract": snapshot_config("printer_v2"),
        "selected_child_count": child_count,
    }
    if full_ingestion:
        plan["selection_mode"] = "all_saved_children"
    args = argparse.Namespace(
        output=tmp_path / "report",
        env_file=tmp_path / "unused.env",
        extract_concurrency=2 if resume_case == "parallel" else 1,
    )
    args.output.mkdir()
    prepared = (
        plan,
        {"B": chosen},
        {"B": saved},
        tokenizer,
        {
            "api_key": "mock",
            "timeout": 20,
            "model": "mock",
            "base_url": "https://mock.invalid",
            "extra_body": {},
        },
        {"api_key": "mock", "dimension": 8, "token_limit": 3072, "model": "mock"},
        {},
    )
    if resume_case in {"over_limit", "invalid_json"}:
        original_review = printer.review_response
        original_build = business_runtime.build_records

        def legacy_build(*args, **kwargs):
            kwargs["on_schema_error"] = "raise"
            return original_build(*args, **kwargs)

        def strict_review(response, sample, **kwargs):
            return original_review(response, sample, record_limit_policy="strict")

        monkeypatch.setattr(printer, "review_response", strict_review)
        monkeypatch.setattr(business_runtime, "build_records", legacy_build)
        assert asyncio.run(runner.run_ingestion(args, prepared)) == 2
        assert "Failed to finalize chunks_vdb" not in caplog.text
        original_summary = (args.output / "summary.json").read_bytes()
        monkeypatch.setattr(printer, "review_response", original_review)
        monkeypatch.setattr(business_runtime, "build_records", original_build)
        args.resume_failed = True
        args.receipt_prefix = "resume_test_"

        def no_new_model_call(*args, **kwargs):
            raise AssertionError(
                "The paid response should be recovered from native cache"
            )

        monkeypatch.setattr(runner, "call_model", no_new_model_call)
        assert asyncio.run(runner.run_ingestion(args, prepared)) == 0
        assert (args.output / "summary.json").read_bytes() == original_summary
    else:
        assert asyncio.run(runner.run_ingestion(args, prepared)) == 0
    prefix = getattr(args, "receipt_prefix", "")
    summary = json.loads(
        (args.output / f"{prefix}summary.json").read_text(encoding="utf-8")
    )
    assert summary["calls"]["extraction_requests"] == (
        0 if resume_case in {"over_limit", "invalid_json"} else child_count
    )
    assert summary["execution"]["extract_concurrency"] == args.extract_concurrency
    if resume_case == "transport":
        assert summary["calls"]["extraction_api_attempts"] == 4
        assert len(list(args.output.glob("request_01_attempt_*_error.json"))) == 3
        assert not (args.output / "response_01.json").exists()
        assert (args.output / "response_02.json").exists()
    if resume_case == "parallel":
        assert peak_active == 2
        for number in (1, 2):
            request = json.loads(
                (args.output / f"request_{number:02d}.json").read_text(encoding="utf-8")
            )
            response_row = json.loads(
                (args.output / f"response_{number:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            assert response_row["request_fingerprint"] == runner.fingerprint(
                request["user_prompt"]
            )
    if resume_case == "invalid_json":
        assert summary["invalidated_cache_entries"] == []
        assert not (args.output / "resume_test_invalidated_cache.json").exists()
    skipped = resume_case in {
        "invalid_json",
        "partial_invalid",
        "missing_section",
        "truncated",
        "transport",
    }
    assert summary["skipped_extraction_chunks"] == int(skipped)
    skipped_rows = json.loads(
        (args.output / f"{prefix}skipped_chunks.json").read_text(encoding="utf-8")
    )
    assert len(skipped_rows) == int(skipped)
    if skipped:
        assert skipped_rows[0]["error"]
    progress = json.loads(
        (args.output / f"{prefix}progress.json").read_text(encoding="utf-8")
    )
    assert progress["stage"] == ("completed_with_skips" if skipped else "completed")
    assert progress["chunks_reviewed"] == progress["chunks_total"] == child_count
    assert progress["skipped_chunks"] == int(skipped)
    assert progress["extraction_percent"] == 100
    assert summary["documents"]["B"]["stored_children"] == child_count
    if full_ingestion:
        assert summary["documents"]["B"]["doc_id"] == chosen[0]["doc_id"]
        assert summary["documents"]["B"]["file_path"] == chosen[0]["document_name"]
        assert summary["documents"]["B"]["stored_parents"] == len(saved.parents)
    chunks = json.loads(
        (args.output / f"{prefix}stored_chunks_B.json").read_text(encoding="utf-8")
    )
    assert len(chunks) == child_count
    row = next(iter(chunks.values()))
    if resume_case == "transport":
        failed = next(
            c
            for c in chunks.values()
            if c["business_extraction"]["review"].get("skipped")
        )
        assert failed["business_extraction_attempt"]["raw_response"] is None
        assert failed["business_extraction_attempt"]["request_failure"]["attempts"] == 3
        assert not failed.get("llm_cache_list")
    assert row["business_extraction"]["facts"] == []
    assert row["business_extraction"]["context"]["parent_id"] == chosen[0]["parent_id"]
    nodes = json.loads(
        (args.output / f"{prefix}graph_nodes_B.json").read_text(encoding="utf-8")
    )
    if resume_case == "over_limit":
        assert len(nodes) == 12
        assert summary["documents"]["B"]["quarantined_records"] == 4
    elif resume_case in {"invalid_json", "missing_section", "truncated"}:
        assert nodes == {}
        assert row["business_extraction"]["review"]["skipped"] is True
        assert row["business_extraction_attempt"]["raw_response"]
    elif resume_case == "parallel":
        assert len(nodes) == 2 and all(
            name.startswith("E-12 ⟦printer:") for name in nodes
        )
    else:
        assert len(nodes) == 1 and next(iter(nodes)).startswith("E-12 ⟦printer:")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transient", "auth", "unexpected"])
async def test_request_retry_policy_keeps_per_attempt_receipts(
    tmp_path, monkeypatch, failure
):
    transient = ExtractionRequestError(
        "Extraction HTTP 503", retryable=True, status=503
    )
    error = {
        "transient": transient,
        "auth": ExtractionRequestError(
            "Extraction HTTP 401", retryable=False, status=401
        ),
        "unexpected": RuntimeError("unexpected bug"),
    }[failure]
    calls = []

    def provider(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(runner, "call_model", provider)
    monkeypatch.setattr(runner, "EXTRACTION_RETRY_DELAYS", (0, 0))
    counters = {"extraction_api_attempts": 0}
    expected = ExtractionRetriesExhausted if failure == "transient" else type(error)
    with pytest.raises(expected):
        await runner.request_extraction(
            {}, [], None, output=tmp_path, stem="request_01", counters=counters
        )
    count = 3 if failure == "transient" else 1
    assert len(calls) == counters["extraction_api_attempts"] == count
    assert len(list(tmp_path.glob("*_error.json"))) == (
        0 if failure == "unexpected" else count
    )


@pytest.mark.asyncio
async def test_transient_request_recovers_without_skipping(tmp_path, monkeypatch):
    provider = AsyncMock(
        side_effect=[
            ExtractionRequestError("Extraction HTTP 429", retryable=True, status=429),
            {"choices": []},
        ]
    )
    monkeypatch.setattr(runner.asyncio, "to_thread", provider)
    sleep = AsyncMock()
    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    counters = {"extraction_api_attempts": 0}
    assert await runner.request_extraction(
        {}, [], None, output=tmp_path, stem="r", counters=counters
    ) == {"choices": []}
    assert counters["extraction_api_attempts"] == 2
    sleep.assert_awaited_once_with(2)


def test_resume_allows_only_implementation_changes_and_keeps_original_ids(tmp_path):
    original = {
        "plan_hash": "old",
        "implementation": {"runtime": "old"},
        "workspace": "current",
    }
    selection = {"B": []}
    (tmp_path / "plan.json").write_text(json.dumps(original), encoding="utf-8")
    (tmp_path / "selected_children.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    current = {**original, "plan_hash": "new", "implementation": {"runtime": "new"}}
    args = argparse.Namespace(output=tmp_path, extract_concurrency=4)
    prepared = runner.resume_prepared(args, (current, selection))
    assert prepared[0]["plan_hash"] == "old"
    assert args.resume_audit["current_implementation"] == {"runtime": "new"}
    with pytest.raises(ValueError, match="workspace"):
        runner.resume_prepared(args, ({**current, "workspace": "other"}, selection))


def test_existing_queue_refuses_before_any_enqueue_or_model_call(tmp_path, monkeypatch):
    class FakeRag:
        def __init__(self, **kwargs):
            self.initialize_storages = AsyncMock()
            self.finalize_storages = AsyncMock()
            self.doc_status = AsyncMock()
            self.doc_status.get_docs_by_statuses.return_value = {
                "existing": {"status": "pending"}
            }
            self.apipeline_enqueue_documents = AsyncMock()
            self.apipeline_process_enqueue_documents = AsyncMock()
            captured.append(self)

    captured = []
    monkeypatch.setattr(runner, "LightRAG", FakeRag)
    monkeypatch.setattr(runner, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(
        runner.socket,
        "create_connection",
        lambda *a, **kw: (_ for _ in ()).throw(OSError()),
    )
    plan = {
        "plan_hash": "queue-test",
        "working_dir": str(tmp_path),
        "workspace": "current",
        "stores": {},
        "selected_child_count": 1,
    }
    args = argparse.Namespace(output=tmp_path, env_file=tmp_path / "unused")
    prepared = (
        plan,
        {"B": [{"document_name": "L5298.pdf"}]},
        {},
        None,
        {
            "api_key": "mock",
            "timeout": 10,
            "model": "mock",
            "base_url": "https://mock.invalid",
            "extra_body": {},
        },
        {"api_key": "mock", "dimension": 8, "token_limit": 3072, "model": "mock"},
        {},
    )
    with pytest.raises(ValueError, match="queued or unfinished"):
        asyncio.run(runner.run_ingestion(args, prepared))
    captured[0].apipeline_enqueue_documents.assert_not_called()
    captured[0].apipeline_process_enqueue_documents.assert_not_called()


def test_full_document_identity_is_stable_and_does_not_use_sample_names():
    selected = {"A": [{"doc_id": "doc-source", "document_name": "L5298.pdf"}]}
    first = runner.document_identities(
        {"selection_mode": "all_saved_children", "plan_hash": "one"}, selected
    )
    second = runner.document_identities(
        {"selection_mode": "all_saved_children", "plan_hash": "two"}, selected
    )
    assert first == second == ({"A": "doc-source"}, {"A": "L5298.pdf"})


def test_sample_cleanup_prevalidates_all_owners_before_deleting():
    rag = AsyncMock()
    rag.full_docs.get_by_id.side_effect = [
        {"file_path": "child_smoke_abcdefgh_L5298.pdf.txt"},
        {"file_path": "unrelated.pdf"},
    ]
    rag.doc_status.get_by_id.side_effect = [
        {"file_path": "child_smoke_abcdefgh_L5298.pdf.txt"},
        {"file_path": "unrelated.pdf"},
    ]
    selected = {
        "A": [{"document_name": "L5298.pdf"}],
        "B": [{"document_name": "L3266.pdf"}],
    }
    with pytest.raises(ValueError, match="ownership"):
        asyncio.run(
            runner.remove_replaced_samples(
                rag, {"replace_sample_plan_hash": "abcdefgh1234"}, selected
            )
        )
    rag.adelete_by_doc_id.assert_not_called()


def test_sample_cleanup_deletes_only_frozen_sample_ids():
    rag = AsyncMock()
    row = {"file_path": "child_smoke_abcdefgh_L5298.pdf.txt"}
    rag.full_docs.get_by_id.return_value = row
    rag.doc_status.get_by_id.return_value = row
    rag.adelete_by_doc_id.return_value.status = "success"
    rag.adelete_by_doc_id.return_value.message = "deleted"
    results = asyncio.run(
        runner.remove_replaced_samples(
            rag,
            {"replace_sample_plan_hash": "abcdefgh1234"},
            {"A": [{"document_name": "L5298.pdf"}]},
        )
    )
    rag.adelete_by_doc_id.assert_awaited_once_with(
        "doc-child-smoke-abcdefgh1234-A", delete_llm_cache=True
    )
    assert results[0]["status"] == "success"


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_progress_retries_windows_locks_without_publishing_partial_json(
    tmp_path, monkeypatch, winerror
):
    path = tmp_path / "progress.json"
    path.write_text('{"chunks_reviewed": 1}', encoding="utf-8")
    original_replace = Path.replace
    attempts, sleeps = [], []

    def replace(source, target):
        attempts.append(source)
        assert json.loads(source.read_text(encoding="utf-8")) == {"chunks_reviewed": 2}
        if len(attempts) < 3:
            assert json.loads(path.read_text(encoding="utf-8")) == {
                "chunks_reviewed": 1
            }
            exc = PermissionError("Temporary sharing violation")
            exc.winerror = winerror
            raise exc
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    runner.write_progress_json(path, {"chunks_reviewed": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"chunks_reviewed": 2}
    assert sleeps == [0.05, 0.1]
    assert len(set(attempts)) == 1
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("winerror", [5, 32, None])
def test_progress_persistent_write_failure_is_visible_and_keeps_previous_snapshot(
    tmp_path, monkeypatch, winerror
):
    path = tmp_path / "progress.json"
    path.write_text('{"chunks_reviewed": 1}', encoding="utf-8")
    attempts, sleeps = [], []

    def replace(source, target):
        attempts.append(source)
        exc = PermissionError("Persistent write denial")
        if winerror is not None:
            exc.winerror = winerror
        raise exc

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError, match="Persistent write denial"):
        runner.write_progress_json(path, {"chunks_reviewed": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"chunks_reviewed": 1}
    assert len(attempts) == (6 if winerror else 1)
    assert sleeps == (list(runner.PROGRESS_RETRY_DELAYS) if winerror else [])
    retained = list(tmp_path.glob("*.tmp"))
    assert len(retained) == 1
    assert retained[0].name != "progress.json.tmp"
    assert json.loads(retained[0].read_text(encoding="utf-8")) == {"chunks_reviewed": 2}


def test_progress_uses_a_different_temporary_file_for_each_publication(
    tmp_path, monkeypatch
):
    paths = []
    original_replace = Path.replace

    def replace(source, target):
        paths.append(source)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    path = tmp_path / "progress.json"
    runner.write_progress_json(path, {"chunks_reviewed": 1})
    runner.write_progress_json(path, {"chunks_reviewed": 2})
    assert len(set(paths)) == 2


def test_progress_counts_cached_chunks_and_seeds_completed_documents(tmp_path, caplog):
    selected = {"A": [{}] * 598, "B": [{}] * 764, "C": [{}] * 619}
    doc_ids = {g: f"doc-{g}" for g in selected}
    counters = {"extraction_requests": 0}
    progress = runner.IngestionProgress(
        {"plan_hash": "progress"},
        selected,
        doc_ids,
        tmp_path / "progress.json",
        counters,
    )
    progress.seed_processed(doc_ids, [{"status": "processed"}, None, None])
    progress.start_document("B")
    record = logging.LogRecord(
        "lightrag", logging.INFO, "", 0, "Business chunk", (), None
    )
    record.printer_chunk_progress = {"doc_id": "doc-B", "completed": 10, "total": 764}
    with caplog.at_level(logging.INFO, logger="lightrag"):
        progress.emit(record)
        progress.emit(record)
    saved = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert saved["chunks_reviewed"] == 608
    assert saved["chunks_total"] == 1981
    assert saved["calls"]["extraction_requests"] == 0
    assert "document B 10/764 chunks; overall 608/1981 chunks (30.7%)" in caplog.text
    record.printer_chunk_progress["completed"] = 764
    progress.emit(record)
    saved = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "extraction_complete_awaiting_ingestion"
    assert saved["chunks_reviewed"] == 1362


def test_cache_recovery_keeps_valid_extractions_and_other_cache_types(tmp_path):
    rag = AsyncMock()
    rag.doc_status.get_by_id.return_value = {
        "status": "failed",
        "chunks_list": ["child"],
    }
    chunk = {"full_doc_id": "doc", "llm_cache_list": ["bad", "good", "keyword"]}
    rag.text_chunks.get_by_ids.return_value = [chunk]
    bad = {"cache_type": "extract", "return": '{"entities": ["missing" "comma"]}'}
    rag.llm_response_cache.get_by_ids.side_effect = [
        [
            bad,
            {"cache_type": "extract", "return": '{"entities":[],"relationships":[]}'},
            {"cache_type": "keywords", "return": "not extraction JSON"},
        ],
        [None],
    ]
    audit = tmp_path / "invalidated_cache.json"
    invalidated = asyncio.run(
        runner.invalidate_malformed_cached_responses(rag, {"A": "doc"}, audit)
    )
    assert invalidated == ["bad"]
    rag.llm_response_cache.delete.assert_awaited_once_with(["bad"])
    rag._flush_storages.assert_awaited_once_with(
        [rag.text_chunks, rag.llm_response_cache]
    )
    assert chunk["llm_cache_list"] == ["bad", "good", "keyword"]
    assert (
        json.loads(audit.read_text(encoding="utf-8"))["entries"]["bad"][
            "original_cache_row"
        ]
        == bad
    )
