"""Attribution survives parent grouping, final budgets and ordinary chunk retrieval."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from lightrag.assistant.model import AssistantFailure
from lightrag.assistant.retrieval import LightRAGAdapter

pytestmark = pytest.mark.offline


async def test_missing_prerequisite_attribution_does_not_get_invented(rag_data):
    rag, rows, _, _ = parent_fixture(rag_data, "child_with_prerequisites")
    rows["doc-1-pc-child-2"]["parent_child"]["context_parent_ids"].append("p2")
    with pytest.raises(AssistantFailure, match="prerequisite"):
        await LightRAGAdapter(rag).retrieve("query")


async def test_ordinary_chunk_keeps_exact_text_and_stable_identity(rag_data):
    rag, _, raw = rag_data
    adapter = LightRAGAdapter(rag)
    first, second = (
        await adapter.retrieve("question"),
        await adapter.retrieve("question"),
    )
    assert first.evidence[0].content == raw["data"]["chunks"][0]["content"]
    assert first.evidence[0].doc_id == "doc-1"
    assert first.evidence[0].evidence_id == second.evidence[0].evidence_id
    assert first.evidence[0].parents == []
    rag.full_docs.get_by_ids.assert_not_awaited()
    assert rag.aquery_data.await_args.kwargs["param"].mode == "mix"


def parent_fixture(rag_data, level="full_parent"):
    rag, rows, raw = rag_data
    rows.clear()
    for i in (1, 2):
        rows[f"doc-1-pc-child-{i}"] = {
            "content": f"child {i}",
            "full_doc_id": "doc-1",
            "file_path": "manual.pdf",
            "parent_child": {
                "doc_id": "doc-1",
                "parent_id": "p1",
                "context_parent_ids": ["p0"],
                "scope": {"applicable_models": ["L3251"]},
            },
        }
    support = list(rows)
    raw["data"]["chunks"] = [
        {
            "chunk_id": support[0],
            "content": "parent with both children and required note",
            "file_path": "manual.pdf",
            "reference_id": "1",
        }
    ]
    raw["metadata"]["parent_child"] = {
        "parents": [
            {
                "doc_id": "doc-1",
                "parent_id": p,
                "source_chunk_id": support[0],
                "source_chunk_ids": support,
                "evidence_level": level,
            }
            for p in ("p1", "p0")
        ]
    }
    raw["metadata"]["retrieval"] = {
        "selected_chunk_ids": [*support, "omitted"],
        "candidate_chunk_ids": [*support, "candidate-only"],
        "budget_omitted_chunk_ids": ["omitted"],
    }
    bundle = {
        "doc_id": "doc-1",
        "parents": {
            p: {
                "doc_id": "doc-1",
                "pages": [2],
                "scope": {
                    "applicable_models": ["L3251"] if p == "p1" else ["L3251", "L4263"]
                },
            }
            for p in ("p1", "p0")
        },
    }
    rag.full_docs.get_by_ids.return_value = [{"parent_child": bundle}]
    return rag, rows, raw, bundle


@pytest.mark.parametrize("level", ["full_parent", "child_with_prerequisites"])
async def test_final_parent_group_retains_all_support_and_local_scopes(rag_data, level):
    rag, _, raw, _ = parent_fixture(rag_data, level)
    batch = await LightRAGAdapter(rag).retrieve("question")
    item = batch.evidence[0]
    assert item.content == raw["data"]["chunks"][0]["content"]
    assert item.source_chunk_ids == ["doc-1-pc-child-1", "doc-1-pc-child-2"]
    assert item.evidence_level == level
    assert item.parents[1].metadata["scope"]["applicable_models"] == ["L3251", "L4263"]
    assert "omitted" not in item.source_chunk_ids
    assert "candidate-only" not in item.source_chunk_ids


@pytest.mark.parametrize(
    "failure",
    [
        "missing_source",
        "different_owner",
        "missing_bundle",
        "missing_parent",
        "missing_mapping",
        "bad_support",
        "unknown_level",
    ],
)
async def test_parent_attribution_fails_loud(rag_data, failure):
    rag, rows, raw, bundle = parent_fixture(rag_data)
    if failure == "missing_source":
        rows.pop("doc-1-pc-child-2")
    elif failure == "different_owner":
        rows["doc-1-pc-child-2"]["full_doc_id"] = "other"
    elif failure == "missing_bundle":
        rag.full_docs.get_by_ids.return_value = [None]
    elif failure == "missing_parent":
        bundle["parents"].pop("p0")
    elif failure == "missing_mapping":
        raw["metadata"].pop("parent_child")
    elif failure == "bad_support":
        raw["metadata"]["parent_child"]["parents"][1]["source_chunk_ids"] = [
            "doc-1-pc-child-1"
        ]
    else:
        for parent in raw["metadata"]["parent_child"]["parents"]:
            parent["evidence_level"] = "unknown"
    with pytest.raises(AssistantFailure) as info:
        await LightRAGAdapter(rag).retrieve("question")
    assert info.value.code == "source_attribution_error"


async def test_confirmed_no_results_is_distinct_from_failed_backend(rag_data):
    rag, _, _ = rag_data
    rag.aquery_data.return_value = None
    rag.aquery_data.side_effect = AsyncMock(
        return_value={"status": "failure", "message": "Query returned empty dataset."}
    )
    assert (await LightRAGAdapter(rag).retrieve("query")).evidence == []
    rag.aquery_data.side_effect = AsyncMock(
        return_value={"status": "failure", "message": "database failed"}
    )
    with pytest.raises(AssistantFailure):
        await LightRAGAdapter(rag).retrieve("query")


async def test_storage_errors_are_not_recast_as_missing_evidence(rag_data):
    rag, _, _ = rag_data
    rag.text_chunks.get_by_ids.side_effect = OSError("backend unavailable")
    with pytest.raises(OSError):
        await LightRAGAdapter(rag).retrieve("query")


async def test_reference_number_is_not_cross_query_identity(rag_data):
    rag, _, raw = rag_data
    adapter = LightRAGAdapter(rag)
    first = (await adapter.retrieve("query")).evidence[0]
    raw["data"]["chunks"][0]["reference_id"] = "8"
    second = (await adapter.retrieve("query")).evidence[0]
    assert first.evidence_id == second.evidence_id
    changed = deepcopy(raw)
    changed["data"]["chunks"][0]["content"] = "different actual source text"
    rag.aquery_data.side_effect = AsyncMock(return_value=changed)
    assert (await adapter.retrieve("query")).evidence[
        0
    ].evidence_id != first.evidence_id
