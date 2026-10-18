"""Rank source children, expand their parents, and select complete evidence groups.

Only persisted source mappings are followed. No model/scene hard filter or
exact-code retrieval channel is implemented here.
"""

from __future__ import annotations

import json

from lightrag.document_structure.table_notes import evidence_text


def candidate_limit(query_param, global_config: dict) -> int:
    """Separate candidate recall from the final chunk limit; zero keeps legacy recall."""
    configured = global_config.get("addon_params", {}).get("rerank_candidate_k", 0)
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 0
    ):
        raise ValueError("rerank_candidate_k must be a non-negative integer")
    return max(configured, query_param.chunk_top_k or query_param.top_k)


async def rank_evidence_children(
    query: str,
    chunks: list[dict],
    query_param,
    global_config: dict,
    chunks_db=None,
    progress_callback=None,
) -> tuple[list[dict], dict]:
    """Rank bounded source children before expanding parents; never filter by model."""
    from lightrag.chunk_schema import format_heading_context
    from lightrag.utils import apply_rerank_if_enabled, logger

    unique, seen = [], set()
    for chunk in chunks:
        key = chunk.get("chunk_id") or chunk.get("id")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(chunk)
    unique = unique[: candidate_limit(query_param, global_config)]
    count = len(unique)
    requested = bool(query_param.enable_rerank)
    enabled = requested
    candidate_ids = [c.get("chunk_id") or c.get("id") for c in unique]
    if enabled and global_config.get("rerank_model_func") is None:
        if global_config.get("addon_params", {}).get("rerank_candidate_k", 0):
            raise ValueError("Rerank requested but no rerank model is configured")
        logger.warning(
            "Rerank requested but no rerank model is configured; retaining original order"
        )
        enabled = False
    if enabled and unique:
        if progress_callback:
            await progress_callback("reranking")
        ids = [c.get("chunk_id") or c.get("id") for c in unique]
        rows = await chunks_db.get_by_ids(ids) if chunks_db else [None] * len(ids)
        inputs = []
        for chunk, row in zip(unique, rows, strict=True):
            meta = (row or {}).get("parent_child") or {}
            heading = meta.get("heading_path") or format_heading_context(row or {})
            scope = meta.get("scope") or {}
            prefix = json.dumps(
                {
                    "document": chunk.get("file_path"),
                    "heading_path": heading,
                    "applicable_models": scope.get("applicable_models", []),
                    "scene": meta.get("feature_domain", "unknown"),
                },
                ensure_ascii=False,
            )
            inputs.append(
                {
                    **chunk,
                    "content": prefix + "\n" + chunk["content"],
                    "_source_content": chunk["content"],
                }
            )
        ranked = await apply_rerank_if_enabled(
            query,
            inputs,
            global_config,
            enable_rerank=True,
            top_n=query_param.chunk_top_k or len(inputs),
        )
        unique = []
        ranked = sorted(ranked, key=lambda c: c.get("rerank_score", 0), reverse=True)
        ranked_ids = set()
        for item in ranked:
            item = dict(item)
            key = item.get("chunk_id") or item.get("id")
            if key and key in ranked_ids:
                continue
            if key:
                ranked_ids.add(key)
            item["content"] = item.pop("_source_content", item.get("content", ""))
            unique.append(item)
        # A provider failure must be visible to experiments, never an unlabelled
        # successful rerank. The generic helper's logged fallback remains usable.
        applied = bool(unique) and all("rerank_score" in c for c in unique)
        if unique and not applied:
            logger.warning("Evidence rerank fell back to original candidate order")
        minimum = global_config.get("min_rerank_score", 0)
        unique = [c for c in unique if c.get("rerank_score", 1) >= minimum]
    else:
        applied = False
    if query_param.chunk_top_k and query_param.chunk_top_k > 0:
        unique = unique[: query_param.chunk_top_k]
    info = {
        "candidate_limit": candidate_limit(query_param, global_config),
        "candidate_count": count,
        "rerank_requested": requested,
        "rerank_applied": applied,
        "selected_child_count": len(unique),
        "selected_chunk_ids": [c.get("chunk_id") or c.get("id") for c in unique],
        "candidate_chunk_ids": candidate_ids,
        "selected_scores": [c.get("rerank_score") for c in unique],
    }
    logger.info(
        "Evidence candidates: %d; selected children: %d; rerank requested=%s applied=%s",
        count,
        len(unique),
        enabled,
        applied,
    )
    return unique, info


def _clean_evidence(chunk: dict, *, fallback: bool = False) -> dict:
    result = {k: v for k, v in chunk.items() if not k.startswith("_parent_")}
    if "_parent_child_ids" in chunk:
        result["supporting_chunk_ids"] = chunk["_parent_child_ids"]
    if fallback:
        result["content"] = chunk["_parent_fallback"]
        result["evidence_level"] = "child_with_prerequisites"
    elif "parent_ids" in chunk:
        result["evidence_level"] = "full_parent"
    return result


def _rendered_size(chunks: list[dict], tokenizer) -> int:
    from lightrag.utils import (
        generate_reference_list_from_chunks,
        render_chunks_context_text,
    )

    _, projected = generate_reference_list_from_chunks(chunks)
    return len(tokenizer.encode(render_chunks_context_text(projected)))


def minimum_evidence_tokens(chunks: list[dict], tokenizer) -> int:
    """Estimate the smallest complete parent/child evidence group, with real rendering."""
    costs = [
        min(
            _rendered_size([_clean_evidence(c)], tokenizer),
            _rendered_size([_clean_evidence(c, fallback=True)], tokenizer)
            if "_parent_fallback" in c
            else float("inf"),
        )
        for c in chunks
    ]
    return min(costs, default=0)


def fit_parent_evidence(
    chunks: list[dict], budget: int, tokenizer
) -> tuple[list[dict], dict]:
    """Keep complete evidence groups; try later candidates when a group cannot fit."""
    result, fallback_ids, omitted_ids = [], [], []
    for chunk in chunks:
        full = _clean_evidence(chunk)
        if budget > 0 and _rendered_size([*result, full], tokenizer) <= budget:
            result.append(full)
        elif "_parent_fallback" in chunk:
            fallback = _clean_evidence(chunk, fallback=True)
            if budget > 0 and _rendered_size([*result, fallback], tokenizer) <= budget:
                result.append(fallback)
                fallback_ids.append(chunk.get("chunk_id"))
            else:
                omitted_ids.append(chunk.get("chunk_id"))
        else:
            omitted_ids.append(chunk.get("chunk_id"))
    return result, {
        "child_fallback_chunk_ids": fallback_ids,
        "budget_omitted_chunk_ids": omitted_ids,
        "final_evidence_count": len(result),
        "evidence_token_budget": budget,
        "evidence_levels": {
            c.get("chunk_id"): c.get("evidence_level", "chunk") for c in result
        },
    }


def retained_parent_metadata(metadata: dict | None, chunks: list[dict]) -> dict | None:
    if metadata is None:
        return None
    retained = {c.get("chunk_id"): c for c in chunks}
    parents = [
        {
            **p,
            "evidence_level": retained[p["source_chunk_id"]].get("evidence_level"),
            "source_chunk_ids": retained[p["source_chunk_id"]].get(
                "supporting_chunk_ids", [p["source_chunk_id"]]
            ),
        }
        for p in metadata["parents"]
        if p["source_chunk_id"] in retained
    ]
    return {**metadata, "parents": parents}


async def expand_parent_evidence(
    chunks: list[dict],
    chunks_db,
    full_docs_db,
    *,
    include_fallback: bool = False,
) -> tuple[list[dict], dict | None]:
    """Deduplicate mapped parents and include their persisted prerequisite parents.

    Keep ordinary chunks unchanged. H evidence must have its original chunk,
    owning document and parent mapping; missing attribution raises an error.
    Final context selection and token budgets remain the caller's responsibility.
    """
    ids = list(
        dict.fromkeys(
            c.get("chunk_id") or c.get("id")
            for c in chunks
            if "-pc-child-" in str(c.get("chunk_id") or c.get("id"))
        )
    )
    if not ids:
        return chunks, None
    if chunks_db is None or full_docs_db is None:
        raise ValueError(
            "Parent evidence requires the current text_chunks and full_docs stores"
        )
    rows = await chunks_db.get_by_ids(ids)
    by_id = {}
    for key, row in zip(ids, rows, strict=True):
        if not row or not row.get("parent_child"):
            raise ValueError("Parent-child source chunk is missing")
        if row["full_doc_id"] != row["parent_child"]["doc_id"]:
            raise ValueError("Parent-child chunk has a different owner")
        by_id[key] = row
    docs = list(dict.fromkeys(row["full_doc_id"] for row in rows))
    doc_rows = await full_docs_db.get_by_ids(docs)
    bundles = {}
    for key, row in zip(docs, doc_rows, strict=True):
        if (
            not row
            or not row.get("parent_child")
            or row["parent_child"]["doc_id"] != key
        ):
            raise ValueError(
                "Parent-child document evidence is missing or has a different owner"
            )
        bundles[key] = row["parent_child"]
    result, seen, sources, groups = [], set(), [], {}
    for chunk in chunks:
        key = chunk.get("chunk_id") or chunk.get("id")
        row = by_id.get(key)
        if row is None:
            result.append(chunk)
            continue
        meta = row["parent_child"]
        bundle = bundles[row["full_doc_id"]]
        group_identity = (row["full_doc_id"], meta["parent_id"])
        if include_fallback and group_identity in groups:
            group = groups[group_identity]
            for parent_id in dict.fromkeys(meta["context_parent_ids"]):
                parent = bundle["parents"].get(parent_id)
                if parent is None or parent["doc_id"] != row["full_doc_id"]:
                    raise ValueError(
                        "Parent-child prerequisite is missing or has a different owner"
                    )
                text = evidence_text(parent)
                if (
                    parent_id != meta["parent_id"]
                    and text not in group["_parent_prerequisites"]
                ):
                    group["_parent_prerequisites"].append(text)
            group["_parent_child_contents"].append(row["content"])
            group["_parent_child_ids"].append(key)
            group["_parent_fallback"] = "\n\n".join(
                [*group["_parent_child_contents"], *group["_parent_prerequisites"]]
            )
            continue
        parent_ids = [meta["parent_id"], *meta["context_parent_ids"]]
        selected, content = [], []
        for parent_id in dict.fromkeys(parent_ids):
            identity = (row["full_doc_id"], parent_id)
            if identity in seen and not include_fallback:
                continue
            parent = bundle["parents"].get(parent_id)
            if parent is None:
                raise ValueError("Parent-child mapping references a missing parent")
            if parent["doc_id"] != row["full_doc_id"]:
                raise ValueError("Parent-child parent has a different owner")
            seen.add(identity)
            content.append(evidence_text(parent))
            selected.append(parent_id)
            sources.append(
                {
                    "doc_id": row["full_doc_id"],
                    "parent_id": parent_id,
                    "source_chunk_id": key,
                    "pages": parent["pages"],
                }
            )
        if content:
            expanded = {
                **chunk,
                "content": "\n\n".join(content),
                "parent_ids": selected,
            }
            if include_fallback:
                prerequisites = [
                    evidence_text(bundle["parents"][p])
                    for p in dict.fromkeys(meta["context_parent_ids"])
                    if p != meta["parent_id"]
                ]
                expanded["_parent_child_contents"] = [row["content"]]
                expanded["_parent_child_ids"] = [key]
                expanded["_parent_prerequisites"] = prerequisites
                expanded["_parent_fallback"] = "\n\n".join(
                    [row["content"], *prerequisites]
                )
                groups[group_identity] = expanded
            result.append(expanded)
    return result, {"version": "parent-child-evidence-v1", "parents": sources}
