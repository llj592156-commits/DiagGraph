"""Native ingestion accepts the business profile independently of the chunker."""

import asyncio
import json

import pytest

from lightrag.base import DocStatus
from lightrag.chunker.parent_child import ParentChildPreparation
from lightrag.extraction import printer
from lightrag.extraction.printer_examples import worked_examples
from tests.pipeline.test_custom_chunking_selector import _force_blocks_path, _new_rag

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("strategy", ["F", "R", "V", "P", "C", "H"])
def test_profile_frozen_at_enqueue_and_uses_existing_stores(
    tmp_path, monkeypatch, strategy
):
    example = worked_examples()[0]
    calls = []
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "100000")

    async def extract(prompt, **kwargs):
        calls.append(kwargs.get("system_prompt", ""))
        return json.dumps(example["output"], ensure_ascii=False)

    def chunks(tokenizer, content, *args, **kwargs):
        return [
            {
                "content": content,
                "tokens": len(tokenizer.encode(content)),
                "chunk_order_index": 0,
            }
        ]

    if strategy == "H":
        import lightrag.chunker.parent_child as module

        def prepare(tokenizer, pdf, path, doc_id, name, **kwargs):
            result = chunks(tokenizer, example["source_text"])
            result[0]["chunk_id"] = "pc-child-one"
            result[0]["parent_child"] = {
                "doc_id": doc_id,
                "parent_id": "parent-one",
                "feature_domain": "network",
                "scope": {
                    "status": "filename",
                    "applicable_models": ["L3266", "L5298"],
                },
                "heading_path": ["网络连接报告"],
                "context_parent_ids": [],
            }
            return ParentChildPreparation(
                result,
                {
                    "version": "test",
                    "doc_id": doc_id,
                    "parents": {
                        "parent-one": {
                            "doc_id": doc_id,
                            "content": example["source_text"],
                            "pages": [1],
                        }
                    },
                },
            )

        monkeypatch.setattr(module, "prepare_parent_child", prepare)
        _force_blocks_path(monkeypatch, tmp_path / "sample.blocks.jsonl")

    async def run():
        options = {
            "extraction_profile": "printer_v2",
            "chunker": {
                "semantic_vector": {
                    "breakpoint_threshold_type": "percentile",
                    "breakpoint_threshold_amount": 100,
                },
            },
        }
        rag = _new_rag(
            tmp_path,
            extract_llm_model_func=extract,
            addon_params=options,
            **({"chunking_func": chunks} if strategy == "C" else {}),
        )
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                example["source_text"],
                ids=["doc-B"],
                file_paths=["L3266_L5298.pdf"],
                process_options=strategy,
            )
            original = await rag.full_docs.get_by_id("doc-B")
            assert original["extraction_config"]["profile"] == "printer_v2"
            # Later config edits cannot switch an already queued document.
            rag.addon_params["extraction_profile"] = "generic"
            if strategy == "H":
                await rag.full_docs.upsert(
                    {"doc-B": {**original, "parse_engine": "docling"}}
                )
                monkeypatch.setattr(
                    rag,
                    "_resolve_source_file_for_parser",
                    lambda *a, **kw: "source.pdf",
                )
            await rag.apipeline_process_enqueue_documents()
            status = await rag.doc_status.get_by_id("doc-B")
            assert DocStatus(status["status"]) is DocStatus.PROCESSED, status
            assert calls == [printer.system_prompt().strip()]
            keys = status["chunks_list"]
            assert len(keys) == 1
            chunk = await rag.text_chunks.get_by_id(keys[0])
            evidence = chunk["business_extraction"]
            assert evidence["context"]["parent_id"] == (
                "parent-one" if strategy == "H" else None
            )
            names = list(evidence["nodes"])
            assert names and all(" ⟦printer:" in name for name in names)
            assert all(
                f["source_node"] in names and f["target_node"] in names
                for f in evidence["facts"]
            )
            for name in names:
                assert await rag.chunk_entity_relation_graph.has_node(name)
            assert await rag.full_entities.get_by_id("doc-B")
            assert await rag.full_relations.get_by_id("doc-B")
            assert chunk["llm_cache_list"]
            await rag.adelete_by_doc_id("doc-B")
            assert await rag.text_chunks.get_by_id(keys[0]) is None
            for name in names:
                assert not await rag.chunk_entity_relation_graph.has_node(name)
        finally:
            await rag.finalize_storages()

    asyncio.run(run())
