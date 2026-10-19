"""Adapt final retrieval evidence without guessing ownership or re-expanding text."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from lightrag.base import QueryParam

from .model import AssistantFailure
from .schemas import EvidenceItem, EvidenceSource, ParentSource, RetrievalBatch


class LightRAGAdapter:
    """Read only final evidence and its persisted attribution in this workspace."""

    def __init__(self, rag, query_param: QueryParam | None = None):
        self.rag = rag
        self.query_param = query_param or QueryParam(
            mode="mix",
            top_k=20,
            chunk_top_k=5,
            max_entity_tokens=2000,
            max_relation_tokens=2000,
            max_total_tokens=8000,
        )
        if self.query_param.mode != "mix":
            raise ValueError("The first assistant version uses mix retrieval")

    async def retrieve(self, query: str) -> RetrievalBatch:
        raw = await self.rag.aquery_data(query, param=deepcopy(self.query_param))
        if not isinstance(raw, dict):
            raise AssistantFailure(
                "retrieval_contract_error",
                "retrieval",
                "Retrieval returned an invalid response",
            )
        if raw.get("status") != "success":
            no_results = raw.get("metadata", {}).get(
                "failure_reason"
            ) == "no_results" or raw.get("message") in {
                "Query returned no results",
                "Query returned empty dataset.",
                "No relevant document chunks found.",
            }
            if no_results:
                return RetrievalBatch(evidence=[], raw_data=raw)
            raise AssistantFailure(
                "retrieval_failed",
                "retrieval",
                "Retrieval did not complete successfully",
            )
        chunks = raw.get("data", {}).get("chunks", [])
        if not isinstance(chunks, list):
            raise AssistantFailure(
                "retrieval_contract_error", "retrieval", "Final chunks must be a list"
            )
        evidence = []
        for chunk in chunks:
            evidence.append(await self._evidence(chunk, raw.get("metadata", {})))
        return RetrievalBatch(evidence=evidence, raw_data=raw)

    async def _evidence(self, chunk: dict, metadata: dict) -> EvidenceItem:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise AssistantFailure("source_attribution_error", "retrieval", message)

        require(isinstance(chunk, dict), "Final evidence must be an object")
        key, content = chunk.get("chunk_id"), chunk.get("content")
        require(
            isinstance(key, str) and bool(key), "Final evidence has no source chunk ID"
        )
        require(
            isinstance(content, str) and bool(content.strip()),
            "Final evidence has no source text",
        )
        mappings = [
            p
            for p in (metadata.get("parent_child") or {}).get("parents", [])
            if p.get("source_chunk_id") == key
        ]
        support = list(
            dict.fromkeys(k for p in mappings for k in p.get("source_chunk_ids", [key]))
        ) or [key]
        require(
            key in support and all(isinstance(k, str) and k for k in support),
            "Invalid supporting source IDs",
        )
        rows = await self.rag.text_chunks.get_by_ids(support)
        require(
            len(rows) == len(support), "Source storage returned an incomplete batch"
        )
        require(
            all(isinstance(r, dict) and r.get("full_doc_id") for r in rows),
            "A final source chunk is missing its owning document",
        )
        docs = {r["full_doc_id"] for r in rows}
        require(len(docs) == 1, "One evidence group cannot mix document owners")
        doc_id = rows[0]["full_doc_id"]
        primary_row = rows[support.index(key)]
        pc = primary_row.get("parent_child")
        sources = []
        for source_id, row in zip(support, rows, strict=True):
            source_pc = row.get("parent_child")
            if source_pc:
                require(
                    source_pc.get("doc_id") == doc_id,
                    "Source parent mapping has a different owner",
                )
            business = (row.get("business_extraction") or {}).get("context")
            sources.append(
                EvidenceSource(
                    chunk_id=source_id,
                    doc_id=doc_id,
                    file_path=row.get("file_path")
                    or chunk.get("file_path")
                    or "unknown_source",
                    parent_child=deepcopy(source_pc),
                    business_context={
                        k: deepcopy(business[k])
                        for k in ("domain", "scope")
                        if k in business
                    }
                    if business
                    else None,
                )
            )
        parents, primary, level = [], None, "chunk"
        if pc or "-pc-child-" in key:
            require(
                bool(pc) and bool(mappings),
                "Final parent evidence has no persisted source mapping",
            )
            primary = pc.get("parent_id")
            require(bool(primary), "Primary parent ID is missing")
            require(
                all(
                    (r.get("parent_child") or {}).get("parent_id") == primary
                    for r in rows
                ),
                "Supporting children must share their primary parent",
            )
            require(
                all(p.get("doc_id") == doc_id for p in mappings),
                "Returned parent metadata has a different document owner",
            )
            require(
                all(
                    set(p.get("source_chunk_ids", [key])) == set(support)
                    for p in mappings
                ),
                "Returned supporting source groups disagree",
            )
            levels = {p.get("evidence_level") for p in mappings}
            require(
                len(levels) == 1
                and next(iter(levels)) in {"full_parent", "child_with_prerequisites"},
                "Final evidence level is unknown",
            )
            level = next(iter(levels))
            doc_rows = await self.rag.full_docs.get_by_ids([doc_id])
            bundle = (
                (doc_rows[0] or {}).get("parent_child") if len(doc_rows) == 1 else None
            )
            require(
                bool(bundle) and bundle.get("doc_id") == doc_id,
                "Owning document parent bundle is missing",
            )
            allowed = {primary}
            for row in rows:
                allowed.update(row["parent_child"].get("context_parent_ids", []))
            parent_ids = list(dict.fromkeys(p.get("parent_id") for p in mappings))
            require(
                primary in parent_ids and set(parent_ids) <= allowed,
                "Returned parents do not belong to the source group",
            )
            require(
                allowed <= set(parent_ids),
                "Final metadata omits a required prerequisite parent",
            )
            for parent_id in parent_ids:
                parent = bundle["parents"].get(parent_id)
                require(
                    bool(parent) and parent.get("doc_id") == doc_id,
                    "A source parent is missing or has a different owner",
                )
                fields = (
                    "pages",
                    "scope",
                    "conditions",
                    "heading_path",
                    "feature_domain",
                    "context_parent_ids",
                )
                parents.append(
                    ParentSource(
                        parent_id=parent_id,
                        doc_id=doc_id,
                        metadata={
                            k: deepcopy(parent[k]) for k in fields if k in parent
                        },
                    )
                )
        else:
            require(
                not mappings and support == [key],
                "Ordinary chunks cannot inherit parent metadata",
            )
        identity = json.dumps(
            [self.rag.workspace, doc_id, sorted(support), level, content],
            ensure_ascii=False,
        )
        return EvidenceItem(
            evidence_id="ev-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            content=content,
            doc_id=doc_id,
            file_path=primary_row.get("file_path")
            or chunk.get("file_path")
            or "unknown_source",
            source_chunk_ids=support,
            sources=sources,
            parents=parents,
            primary_parent_id=primary,
            evidence_level=level,
            original_reference_id=str(chunk.get("reference_id") or ""),
        )
