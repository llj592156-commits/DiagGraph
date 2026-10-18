"""H uses the existing pipeline and graph extractor; all API calls are mocked."""

import asyncio
import json

import pytest

from lightrag import QueryParam
from lightrag.base import DocStatus
from lightrag.chunker.parent_child import VERSION, ParentChildPreparation
from tests.pipeline.test_custom_chunking_selector import _force_blocks_path, _new_rag

pytestmark = pytest.mark.offline

BODY = "E-12 is a network error. Check the network connection."
EXTRACTED = (
    "entity<|#|>E-12<|#|>CONCEPT<|#|>A network error\n"
    "entity<|#|>Network<|#|>CONCEPT<|#|>The network connection\n"
    "relation<|#|>E-12<|#|>Network<|#|>network<|#|>Check the connection\n"
    "<|COMPLETE|>"
)


@pytest.mark.parametrize("hard_limit", [None, 35])
def test_h_retains_native_graph_and_maps_final_children_to_parents(
    tmp_path, monkeypatch, hard_limit
):
    import lightrag.chunker.parent_child as chunker

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "100000")
    monkeypatch.setenv("EXTRACTION_PROFILE", "generic")
    calls, captured = [], {}

    async def extract(prompt, **kwargs):
        calls.append(kwargs.get("system_prompt", ""))
        return EXTRACTED

    def prepare(tokenizer, pdf, blocks_path, doc_id, document_name, **options):
        captured.update(options)
        meta = {
            "doc_id": doc_id,
            "parent_id": "p1",
            "child_id": "child-one",
            "context_parent_ids": [],
            "version": VERSION,
        }
        parent = {
            "doc_id": doc_id,
            "content": BODY + "\nAdditional prerequisite from the full parent.",
            "table_notes": [],
            "pages": [234],
        }
        return ParentChildPreparation(
            [
                {
                    "chunk_id": "pc-child-one",
                    "content": BODY,
                    "tokens": len(tokenizer.encode(BODY)),
                    "chunk_order_index": 0,
                    "parent_child": meta,
                }
            ],
            {"doc_id": doc_id, "parents": {"p1": parent}, "version": VERSION},
        )

    monkeypatch.setattr(chunker, "prepare_parent_child", prepare)
    _force_blocks_path(monkeypatch, tmp_path / "input.blocks.jsonl")

    async def run():
        rag = _new_rag(
            tmp_path,
            extract_llm_model_func=extract,
            entity_extract_max_gleaning=0,
            entity_extraction_use_json=False,
        )
        await rag.initialize_storages()
        try:
            if hard_limit:
                rag.embedding_token_limit = hard_limit
                rag.embedding_chunk_overlap_token_size = 5
            await rag.apipeline_enqueue_documents(
                BODY, ids=["doc-B"], file_paths=["L3266_L5298.pdf"], process_options="H"
            )
            record = await rag.full_docs.get_by_id("doc-B")
            await rag.full_docs.upsert({"doc-B": {**record, "parse_engine": "docling"}})
            monkeypatch.setattr(
                rag,
                "_resolve_source_file_for_parser",
                lambda *_args, **_kwargs: "source.pdf",
            )
            await rag.apipeline_process_enqueue_documents()
            status = await rag.doc_status.get_by_id("doc-B")
            assert DocStatus(status["status"]) is DocStatus.PROCESSED, status
            assert status["metadata"]["chunk_method"] == "parent_child"
            assert captured == {"chunk_token_size": 500, "chunk_overlap_token_size": 60}
            doc = await rag.full_docs.get_by_id("doc-B")
            keys = status["chunks_list"]
            assert set(
                doc["parent_child"]["parents"]["p1"]["indexed_chunk_ids"]
            ) == set(keys)
            assert len(keys) > 1 if hard_limit else len(keys) == 1
            for key in keys:
                chunk = await rag.text_chunks.get_by_id(key)
                assert chunk["parent_child"]["parent_id"] == "p1"
                assert "parent_child_kg" not in chunk
                assert chunk["llm_cache_list"]
            assert await rag.chunk_entity_relation_graph.has_node("E-12")
            assert await rag.chunk_entity_relation_graph.has_node("Network")
            assert calls and all(
                "printer-business-extraction" not in prompt for prompt in calls
            )
            # Exercise the real existing five query modes with no keyword/answer calls.
            for mode in ["naive", "local", "global", "hybrid", "mix"]:
                data = await rag.aquery_data(
                    "E-12 network",
                    QueryParam(
                        mode=mode,
                        enable_rerank=False,
                        hl_keywords=["network"],
                        ll_keywords=["E-12"],
                    ),
                )
                assert "Additional prerequisite" in json.dumps(data), (mode, data)
                assert (
                    data["metadata"]["parent_child"]["parents"][0]["parent_id"] == "p1"
                )
            await rag.adelete_by_doc_id("doc-B")
            assert await rag.full_docs.get_by_id("doc-B") is None
            for key in keys:
                assert await rag.text_chunks.get_by_id(key) is None
            assert not await rag.chunk_entity_relation_graph.has_node("E-12")
        finally:
            await rag.finalize_storages()

    asyncio.run(run())


@pytest.mark.parametrize(
    "engine,filename", [("legacy", "guide.pdf"), ("docling", "guide.txt")]
)
def test_h_fails_instead_of_using_fixed_chunk_fallback(
    tmp_path, monkeypatch, engine, filename
):
    async def run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                BODY, ids=["bad-H"], file_paths=[filename], process_options="H!"
            )
            record = await rag.full_docs.get_by_id("bad-H")
            await rag.full_docs.upsert({"bad-H": {**record, "parse_engine": engine}})
            await rag.apipeline_process_enqueue_documents()
            status = await rag.doc_status.get_by_id("bad-H")
            assert DocStatus(status["status"]) is DocStatus.FAILED
            assert "requires a PDF parsed with Docling" in status["error_msg"]
            assert not status.get("chunks_list")
        finally:
            await rag.finalize_storages()

    asyncio.run(run())
