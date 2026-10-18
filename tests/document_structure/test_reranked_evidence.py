"""Bounded child reranking and complete parent evidence use no live providers."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from lightrag.base import QueryParam
from lightrag.document_structure.query import (
    candidate_limit,
    expand_parent_evidence,
    fit_parent_evidence,
    minimum_evidence_tokens,
    rank_evidence_children,
    retained_parent_metadata,
)
from lightrag.operate import _build_context_str, naive_query

pytestmark = pytest.mark.offline


class CharTokenizer:
    def encode(self, text):
        return list(text)


def sources():
    rows = {
        "doc-B-pc-child-one": {
            "full_doc_id": "doc-B",
            "content": "Complete child. Do not use alcohol.",
            "parent_child": {
                "doc_id": "doc-B",
                "parent_id": "p1",
                "context_parent_ids": ["p0"],
                "heading_path": ["Maintenance"],
                "scope": {"applicable_models": ["L5298"]},
            },
        },
    }
    parents = {
        "p1": {"doc_id": "doc-B", "content": "Long parent " * 2000, "pages": [2]},
        "p0": {"doc_id": "doc-B", "content": "Disconnect power first.", "pages": [1]},
    }
    chunks, docs = AsyncMock(), AsyncMock()
    chunks.get_by_ids.side_effect = lambda ids: [deepcopy(rows.get(key)) for key in ids]
    docs.get_by_ids.return_value = [
        {"parent_child": {"doc_id": "doc-B", "parents": parents}}
    ]
    hits = [
        {"chunk_id": key, "content": row["content"], "file_path": "L5298.pdf"}
        for key, row in rows.items()
    ]
    return chunks, docs, rows, parents, hits


@pytest.mark.asyncio
async def test_twenty_candidates_reranked_to_five_without_model_filtering():
    hits = [
        {"chunk_id": f"chunk-{i}", "content": f"original {i}", "file_path": "L5298.pdf"}
        for i in range(30)
    ]
    chunks = AsyncMock()
    chunks.get_by_ids.side_effect = lambda ids: [
        {
            "parent_child": {
                "heading_path": ["Guide"],
                "scope": {"applicable_models": ["L5298"]},
            }
        }
        for _ in ids
    ]
    provider = AsyncMock(
        return_value=[{"index": i, "relevance_score": 0.9} for i in (19, 4, 3, 2, 1)]
    )
    config = {
        "addon_params": {"rerank_candidate_k": 20},
        "rerank_model_func": provider,
        "min_rerank_score": 0,
    }
    selected, info = await rank_evidence_children(
        "L3266 question",
        hits,
        QueryParam(chunk_top_k=5, enable_rerank=True),
        config,
        chunks,
    )
    args = provider.await_args.kwargs
    assert len(args["documents"]) == 20 and args["top_n"] == 5
    assert "L5298" in args["documents"][0] and "Guide" in args["documents"][0]
    assert [c["chunk_id"] for c in selected] == [
        "chunk-19",
        "chunk-4",
        "chunk-3",
        "chunk-2",
        "chunk-1",
    ]
    assert selected[0]["content"] == "original 19"
    assert hits[0]["content"] == "original 0"
    assert info["candidate_count"] == 20 and info["rerank_applied"] is True


@pytest.mark.asyncio
async def test_rerank_failure_remains_visible_in_result_metadata():
    provider = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    selected, info = await rank_evidence_children(
        "query",
        [{"chunk_id": "one", "content": "source"}],
        QueryParam(enable_rerank=True),
        {"addon_params": {"rerank_candidate_k": 20}, "rerank_model_func": provider},
    )
    assert selected[0]["content"] == "source"
    assert info["rerank_applied"] is False
    assert info["rerank_requested"] is True


@pytest.mark.asyncio
async def test_requested_rerank_cannot_silently_run_without_a_model():
    with pytest.raises(ValueError, match="no rerank model"):
        await rank_evidence_children(
            "query",
            [{"content": "source"}],
            QueryParam(enable_rerank=True),
            {"addon_params": {"rerank_candidate_k": 20}},
        )


@pytest.mark.asyncio
async def test_long_parent_falls_back_to_complete_child_and_prerequisites():
    chunks, docs, _, _, hits = sources()
    expanded, metadata = await expand_parent_evidence(
        hits, chunks, docs, include_fallback=True
    )
    tokenizer = CharTokenizer()
    budget = minimum_evidence_tokens(expanded, tokenizer)
    retained, info = fit_parent_evidence(expanded, budget, tokenizer)
    assert len(retained) == 1
    assert hits[0]["content"] in retained[0]["content"]
    assert "Disconnect power first." in retained[0]["content"]
    assert retained[0]["evidence_level"] == "child_with_prerequisites"
    assert info["child_fallback_chunk_ids"] == [hits[0]["chunk_id"]]
    assert (
        retained_parent_metadata(metadata, retained)["parents"][0]["evidence_level"]
        == "child_with_prerequisites"
    )
    assert not any(key.startswith("_parent_") for key in retained[0])


@pytest.mark.asyncio
async def test_parent_group_fallback_keeps_both_selected_children():
    chunks, docs, rows, _, hits = sources()
    rows["doc-B-pc-child-two"] = deepcopy(rows[hits[0]["chunk_id"]])
    rows["doc-B-pc-child-two"]["content"] = "Second complete action."
    hits.append(
        {
            "chunk_id": "doc-B-pc-child-two",
            "content": "Second complete action.",
            "file_path": "L5298.pdf",
        }
    )
    expanded, _ = await expand_parent_evidence(
        hits, chunks, docs, include_fallback=True
    )
    assert len(expanded) == 1
    retained, _ = fit_parent_evidence(expanded, 500, CharTokenizer())
    assert "Second complete action." in retained[0]["content"]
    assert hits[0]["content"] in retained[0]["content"]


@pytest.mark.asyncio
async def test_prerequisite_that_cannot_fit_is_not_silently_clipped():
    chunks, docs, _, parents, hits = sources()
    parents["p0"]["content"] = "Critical prerequisite " * 1000
    expanded, metadata = await expand_parent_evidence(
        hits, chunks, docs, include_fallback=True
    )
    retained, info = fit_parent_evidence(expanded, 500, CharTokenizer())
    assert retained == [] and info["budget_omitted_chunk_ids"]
    assert retained_parent_metadata(metadata, retained)["parents"] == []


def test_oversized_first_candidate_does_not_hide_later_fitting_evidence():
    chunks = [
        {"chunk_id": "big", "content": "a" * 5000},
        {"chunk_id": "small", "content": "Complete useful source."},
    ]
    retained, info = fit_parent_evidence(chunks, 300, CharTokenizer())
    assert [c["chunk_id"] for c in retained] == ["small"]
    assert info["budget_omitted_chunk_ids"] == ["big"]


@pytest.mark.asyncio
async def test_kg_summaries_yield_budget_to_source_evidence():
    chunks, docs, _, _, hits = sources()
    config = {"tokenizer": CharTokenizer(), "addon_params": {"rerank_candidate_k": 20}}
    context, raw = await _build_context_str(
        [{"entity": str(i), "description": "description " * 100} for i in range(8)],
        [],
        hits,
        "query",
        QueryParam(
            mode="hybrid", enable_rerank=False, chunk_top_k=5, max_total_tokens=2500
        ),
        config,
        system_prompt="Context: {context_data}",
        text_chunks_db=chunks,
        full_docs_db=docs,
    )
    assert hits[0]["content"] in context
    assert "Disconnect power first." in context
    assert len(raw["data"]["chunks"]) == 1
    assert len(raw["data"]["entities"]) < 8
    assert raw["metadata"]["retrieval"]["child_fallback_chunk_ids"]
    assert len(context) + len("Context: ") + len("query") <= 2500


@pytest.mark.asyncio
async def test_naive_recall_uses_twenty_and_reranks_children_before_parent_lookup():
    chunks, docs, _, _, hits = sources()
    many = [
        {"chunk_id": f"chunk-{i}", "content": f"source {i}", "file_path": "plain.txt"}
        for i in range(19)
    ] + hits
    provider = AsyncMock(return_value=[{"index": 19, "relevance_score": 0.9}])
    config = {
        "tokenizer": CharTokenizer(),
        "addon_params": {"rerank_candidate_k": 20},
        "rerank_model_func": provider,
        "role_llm_funcs": {"query": AsyncMock()},
        "min_rerank_score": 0,
    }
    vdb = AsyncMock()
    vdb.global_config = config
    vdb.cosine_better_than_threshold = 0.2
    vdb.query.return_value = [{**c, "id": c["chunk_id"]} for c in many]
    result = await naive_query(
        "query",
        vdb,
        QueryParam(
            mode="naive",
            chunk_top_k=5,
            enable_rerank=True,
            only_need_context=True,
            max_total_tokens=5000,
        ),
        config,
        text_chunks_db=chunks,
        full_docs_db=docs,
    )
    assert vdb.query.await_args.kwargs["top_k"] == 20
    assert hits[0]["content"] in provider.await_args.kwargs["documents"][19]
    assert "Long parent " not in provider.await_args.kwargs["documents"][19]
    assert result.raw_data["metadata"]["retrieval"]["rerank_applied"] is True
    assert len(result.raw_data["data"]["chunks"]) == 1


def test_recall_defaults_remain_compatible_without_candidate_override():
    param = QueryParam(top_k=20, chunk_top_k=5)
    assert candidate_limit(param, {}) == 5
    assert candidate_limit(param, {"addon_params": {"rerank_candidate_k": 20}}) == 20


@pytest.mark.asyncio
async def test_hybrid_merge_caps_shared_candidate_pool_after_source_dedup(monkeypatch):
    from lightrag import operate

    def hits(prefix):
        return [
            {"chunk_id": f"{prefix}-{i}", "content": f"text {i}"} for i in range(20)
        ]

    monkeypatch.setattr(
        operate,
        "_find_related_text_unit_from_entities",
        AsyncMock(return_value=hits("entity")),
    )
    monkeypatch.setattr(
        operate,
        "_find_related_text_unit_from_relations",
        AsyncMock(return_value=hits("relation")),
    )
    store = AsyncMock()
    store.global_config = {"addon_params": {"rerank_candidate_k": 20}}
    merged = await operate._merge_all_chunks(
        [{}],
        [{}],
        hits("vector"),
        text_chunks_db=store,
        query_param=QueryParam(chunk_top_k=5),
    )
    assert len(merged) == 20
    assert [c["chunk_id"] for c in merged[:6]] == [
        "vector-0",
        "entity-0",
        "relation-0",
        "vector-1",
        "entity-1",
        "relation-1",
    ]


@pytest.mark.asyncio
async def test_rerank_duplicate_indices_keep_best_unique_source():
    provider = AsyncMock(
        return_value=[
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.7},
            {"index": 0, "relevance_score": 0.9},
        ]
    )
    selected, _ = await rank_evidence_children(
        "query",
        [{"chunk_id": "one", "content": "one"}, {"chunk_id": "two", "content": "two"}],
        QueryParam(chunk_top_k=5, enable_rerank=True),
        {"addon_params": {"rerank_candidate_k": 20}, "rerank_model_func": provider},
    )
    assert [c["chunk_id"] for c in selected] == ["one", "two"]
    assert selected[0]["rerank_score"] == 0.9


@pytest.mark.asyncio
async def test_later_group_keeps_shared_prerequisite_when_earlier_group_cannot_fit():
    chunks, docs, rows, parents, hits = sources()
    rows["doc-B-pc-child-two"] = deepcopy(rows[hits[0]["chunk_id"]])
    rows["doc-B-pc-child-two"]["parent_child"]["parent_id"] = "p2"
    rows[hits[0]["chunk_id"]]["content"] = "Large required child " * 1000
    rows["doc-B-pc-child-two"]["content"] = "Second child."
    parents["p2"] = {"doc_id": "doc-B", "content": "Small parent.", "pages": [3]}
    hits.append(
        {
            "chunk_id": "doc-B-pc-child-two",
            "content": "Second child.",
            "file_path": "L5298.pdf",
        }
    )
    expanded, _ = await expand_parent_evidence(
        hits, chunks, docs, include_fallback=True
    )
    retained, info = fit_parent_evidence(expanded, 250, CharTokenizer())
    assert len(retained) == 1 and retained[0]["chunk_id"] == "doc-B-pc-child-two"
    assert "Disconnect power first." in retained[0]["content"]
    assert info["budget_omitted_chunk_ids"] == [hits[0]["chunk_id"]]
