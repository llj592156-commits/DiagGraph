"""Preflight or ingest complete saved children into the current knowledge base.

--run writes the CURRENT configured knowledge base and calls extraction and
embedding APIs. Stop other writers first. No PDF parsing or re-chunking runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values, load_dotenv

from lightrag import LightRAG, RoleLLMConfig
from lightrag.base import DocStatus
from lightrag.chunker import accepts_chunking_context
from lightrag.chunker.parent_child import child_evidence_content
from lightrag.document_structure.children import load_local_tokenizer
from lightrag.document_structure.parents import file_hash
from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.evaluation.batch_baseline import write_json
from lightrag.evaluation.compare_parent_child_retrieval import (
    EMBEDDING_REQUEST_TIMEOUT,
    EMBEDDING_RETRY_DELAYS,
    embed_batch,
    embedding_config,
    experiment_lock,
    provider_identity,
)
from lightrag.evaluation.printer_graph_smoke import call_model, config, public_config
from lightrag.extraction import printer
from lightrag.extraction.errors import (
    ExtractionRequestError,
    ExtractionRetriesExhausted,
)
from lightrag.extraction.runtime import fingerprint, snapshot_config
from lightrag.utils import EmbeddingFunc, TruncatedResponse, logger
from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result

VERSION = "printer-saved-child-ingestion-v1"
EXTRACTION_RETRY_DELAYS = (2, 4)
PROGRESS_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
STORES = {
    "kv_storage": ("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
    "vector_storage": ("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
    "graph_storage": ("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
    "doc_status_storage": ("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
}


def write_progress_json(path: Path, value: dict) -> None:
    """Publish progress atomically; retry brief Windows replacement locks.

    A persistent failure raises and retains the previous snapshot plus the
    unique temporary file for inspection. Neither is an ingestion checkpoint.
    """
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    for attempt in range(len(PROGRESS_RETRY_DELAYS) + 1):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(
                PROGRESS_RETRY_DELAYS
            ):
                raise
            if attempt == 0:
                logger.warning(
                    "Progress snapshot %s is temporarily locked; retrying publication",
                    path,
                )
            time.sleep(PROGRESS_RETRY_DELAYS[attempt])


class IngestionProgress(logging.Handler):
    """Count reviewed source chunks, including cache hits, separately from API calls."""

    def __init__(self, plan, selected, doc_ids, output, counters):
        super().__init__()
        self.plan_hash = plan["plan_hash"]
        self.expected = {g: len(rows) for g, rows in selected.items()}
        self.completed = dict.fromkeys(selected, 0)
        self.skipped = {g: set() for g in selected}
        self.groups = {doc_id: group for group, doc_id in doc_ids.items()}
        self.output = output
        self.counters = counters
        self.stage = "starting"
        self.active_group = None
        self.last_response = None

    def save(self):
        total = sum(self.expected.values())
        completed = sum(self.completed.values())
        write_progress_json(
            self.output,
            {
                "plan_hash": self.plan_hash,
                "stage": self.stage,
                "active_document": self.active_group,
                "chunks_reviewed": completed,
                "chunks_total": total,
                "skipped_chunks": sum(len(rows) for rows in self.skipped.values()),
                "extraction_percent": round(100 * completed / total, 1) if total else 0,
                "documents": {
                    g: {
                        "chunks_reviewed": self.completed[g],
                        "chunks_total": count,
                        "skipped_chunks": len(self.skipped[g]),
                    }
                    for g, count in self.expected.items()
                },
                "calls": dict(self.counters),
                "last_response": self.last_response,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def seed_processed(self, doc_ids, rows):
        for (group, _), row in zip(doc_ids.items(), rows, strict=True):
            if row and row.get("status") == DocStatus.PROCESSED.value:
                self.completed[group] = self.expected[group]
        self.save()

    def start_document(self, group):
        self.active_group = group
        self.completed[group] = 0
        self.skipped[group].clear()
        self.stage = "extracting"
        self.save()
        logger.info(
            "Extraction started: document %s 0/%d chunks; overall %d/%d chunks",
            group,
            self.expected[group],
            sum(self.completed.values()),
            sum(self.expected.values()),
        )

    def emit(self, record):
        event = getattr(record, "printer_chunk_progress", None)
        if not event or event["doc_id"] not in self.groups:
            return
        group = self.groups[event["doc_id"]]
        if (
            event["total"] != self.expected[group]
            or not 0 <= event["completed"] <= event["total"]
        ):
            raise ValueError("Extraction progress does not match the saved child plan")
        self.active_group = group
        self.completed[group] = event["completed"]
        if event.get("skipped"):
            self.skipped[group].add(event["chunk_id"])
        complete = event["completed"] == event["total"]
        self.stage = (
            "extraction_complete_awaiting_ingestion" if complete else "extracting"
        )
        self.save()
        completed, total = sum(self.completed.values()), sum(self.expected.values())
        logger.info(
            "Extraction progress: document %s %d/%d chunks; overall %d/%d chunks (%.1f%%); skipped %d chunks%s",
            group,
            event["completed"],
            event["total"],
            completed,
            total,
            100 * completed / total,
            sum(len(rows) for rows in self.skipped.values()),
            "; document extraction complete, graph merge and vector indexing still required"
            if complete
            else "",
        )


def select_children(index: ParentChildIndex, count: int) -> list[dict]:
    """Pick complete existing children for coverage; never crop or synthesize text."""
    children = list(index.children.values())
    chosen = []
    predicates = [
        lambda c: c["feature_domain"] == "network" and "E-12" in c["content"],
        lambda c: (
            c["feature_domain"] == "device_display_errors" and "E-12" in c["content"]
        ),
        lambda c: (
            c["feature_domain"] == "device_display_errors" and "W-11" in c["content"]
        ),
        lambda c: "仅对于" in c["content"] or "仅适用" in c["content"],
    ]
    for predicate in predicates:
        match = next((c for c in children if c not in chosen and predicate(c)), None)
        if match is not None and len(chosen) < count:
            chosen.append(match)
    ranked = sorted(
        children,
        key=lambda c: sum(
            word in c["content"] for word in ("故障", "检查", "解决", "不支持", "规格")
        ),
        reverse=True,
    )
    for child in ranked:
        if len(chosen) >= count:
            break
        if child not in chosen:
            chosen.append(child)
    if len(chosen) != count:
        raise ValueError("Not enough saved children for the requested sample count")
    return deepcopy(chosen)


def prepare(args):
    values = {**dotenv_values(args.env_file), **os.environ}
    indexes = {
        group: ParentChildIndex.from_directory(args.bundles / group) for group in "ABC"
    }
    all_children = getattr(args, "all_children", False)
    selected = {
        group: (
            deepcopy(list(index.children.values()))
            if all_children
            else select_children(index, args.per_document)
        )
        for group, index in indexes.items()
    }
    tokenizer = load_local_tokenizer(args.tokenizer)
    extract = config(args.env_file, args.max_output_tokens)
    embedding = embedding_config(args.env_file)
    stores = {
        key: str(values.get(env) or default) for key, (env, default) in STORES.items()
    }
    if any(stores[key] != default for key, (_, default) in STORES.items()):
        raise ValueError(
            "This bounded runner supports the currently used JSON/NetworkX/NanoVectorDB backends only"
        )
    sizes = [
        len(tokenizer.encode(child_evidence_content(c)))
        for rows in selected.values()
        for c in rows
    ]
    if max(sizes) > embedding["token_limit"]:
        raise ValueError(
            "A complete selected child exceeds the embedding limit; refusing re-splitting"
        )
    plan = {
        "version": VERSION,
        "contract": snapshot_config("printer_v2"),
        "working_dir": str(Path(values.get("WORKING_DIR") or "rag_storage").resolve()),
        "workspace": str(values.get("WORKSPACE") or ""),
        "stores": stores,
        "extract": public_config(extract),
        "embedding": provider_identity(embedding),
        "inputs": {g: file_hash(args.bundles / g / "summary.json") for g in indexes},
        "children": {
            g: [{"child_id": c["child_id"], "sha256": fingerprint(c)} for c in rows]
            for g, rows in selected.items()
        },
        "implementation": {
            p: file_hash(Path(p))
            for p in (
                __file__,
                "lightrag/extraction/runtime.py",
                "lightrag/extraction/errors.py",
                "lightrag/evaluation/printer_graph_smoke.py",
                "lightrag/evaluation/compare_parent_child_retrieval.py",
                "lightrag/extraction/printer.py",
                "lightrag/chunker/parent_child.py",
            )
        },
        "selected_child_count": len(sizes),
        "max_child_input_tokens": max(sizes),
        "answer_requests": 0,
        "keyword_requests": 0,
        "parsing_requests": 0,
    }
    if all_children:
        plan["selection_mode"] = "all_saved_children"
        plan["parent_counts"] = {g: len(index.parents) for g, index in indexes.items()}
        replacement = getattr(args, "replace_sample_run", None)
        if replacement is not None:
            sample_plan = json.loads(
                (replacement / "plan.json").read_text(encoding="utf-8")
            )
            if any(
                sample_plan[key] != plan[key]
                for key in ("working_dir", "workspace", "stores")
            ):
                raise ValueError(
                    "Replacement samples belong to a different knowledge base"
                )
            plan["replace_sample_plan_hash"] = sample_plan["plan_hash"]
    plan["plan_hash"] = fingerprint(plan)
    return plan, selected, indexes, tokenizer, extract, embedding, values


def prepared_chunks(rows, index, doc_id, name, tokenizer, *, include_all_parents=False):
    parents, chunks = {}, []

    def add_parent(key):
        if key in parents:
            return
        parent = deepcopy(index.parents[key])
        parent["source_doc_id"] = parent["doc_id"]
        parent["doc_id"] = doc_id
        parent["indexed_chunk_ids"] = []
        parents[key] = parent
        for ancestor in parent["context_parent_ids"]:
            add_parent(ancestor)

    if include_all_parents:
        for key in index.parents:
            add_parent(key)
    for order, child in enumerate(rows):
        add_parent(child["parent_id"])
        metadata = deepcopy(child)
        metadata.pop("content")
        metadata.pop("retrieval_text", None)
        metadata["source_doc_id"] = metadata["doc_id"]
        metadata["doc_id"] = doc_id
        content = child_evidence_content(child)
        chunks.append(
            {
                "chunk_id": "pc-" + child["child_id"],
                "content": content,
                "tokens": len(tokenizer.encode(content)),
                "chunk_order_index": order,
                "heading": {
                    "heading": child["heading_path"][-1],
                    "parent_headings": child["heading_path"][:-1],
                },
                "parent_child": metadata,
            }
        )
    final = build_chunks_dict_from_chunking_result(
        chunks, doc_id=doc_id, file_path=name
    )
    for key, chunk in final.items():
        parents[chunk["parent_child"]["parent_id"]]["indexed_chunk_ids"].append(key)
    return chunks, {"version": VERSION, "doc_id": doc_id, "parents": parents}


def document_identities(plan, selected):
    """Use original saved document identities for full ingestion, sample IDs otherwise."""
    if plan.get("selection_mode") == "all_saved_children":
        for rows in selected.values():
            if not rows or len({(c["doc_id"], c["document_name"]) for c in rows}) != 1:
                raise ValueError(
                    "Full ingestion requires one owning document per bundle"
                )
        return (
            {g: rows[0]["doc_id"] for g, rows in selected.items()},
            {g: rows[0]["document_name"] for g, rows in selected.items()},
        )
    return (
        {g: f"doc-child-smoke-{plan['plan_hash'][:12]}-{g}" for g in selected},
        {
            g: f"child_smoke_{plan['plan_hash'][:8]}_{rows[0]['document_name']}.txt"
            for g, rows in selected.items()
        },
    )


async def remove_replaced_samples(rag, plan, selected):
    """Delete known sample documents only after every complete manual passed verification."""
    sample_hash = plan.get("replace_sample_plan_hash")
    if not sample_hash:
        return []
    verified = []
    for group in selected:
        doc_id = f"doc-child-smoke-{sample_hash[:12]}-{group}"
        row = await rag.full_docs.get_by_id(doc_id)
        status = await rag.doc_status.get_by_id(doc_id)
        if row is None and status is None:
            continue
        expected = (
            f"child_smoke_{sample_hash[:8]}_{selected[group][0]['document_name']}.txt"
        )
        if (
            not row
            or not status
            or row.get("file_path") != expected
            or status.get("file_path") != expected
        ):
            raise ValueError("Replacement sample ownership could not be verified")
        verified.append(doc_id)
    results = []
    for doc_id in verified:
        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        results.append(
            {"doc_id": doc_id, "status": result.status, "message": result.message}
        )
        if result.status != "success":
            raise RuntimeError(
                f"Could not delete replaced sample {doc_id}: {result.message}"
            )
    return results


async def publish_sample_retry(rag, doc_ids, names, plan):
    """Use normal sticky manual retry only when every failed doc is ours."""
    from lightrag.kg.pipeline_ingress import (
        ManualRetryPublishResult,
        PipelineIngressMessage,
    )
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_namespace_lock,
        get_pipeline_ingress,
    )

    failed = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED], strict=True)
    if not set(failed).issubset(set(doc_ids.values())):
        raise ValueError(
            "Other failed documents exist; refusing a workspace-wide manual retry"
        )
    for group, doc_id in doc_ids.items():
        row = await rag.full_docs.get_by_id(doc_id)
        if (
            not row
            or row.get("file_path") != names[group]
            or row.get("extraction_config") != plan["contract"]
        ):
            raise ValueError(
                "Resume document ownership or extraction contract differs from frozen plan"
            )
    if not failed:
        return
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    ingress = await get_pipeline_ingress(workspace=rag.workspace)
    request_id = f"child-smoke-resume-{uuid4().hex}"
    async with get_namespace_lock("pipeline_status", workspace=rag.workspace):
        if (
            status.get("busy")
            or status.get("scanning")
            or ingress.counts()["manual_retries"]
        ):
            raise ValueError("A pipeline/manual retry operation is already active")
        result = ingress.request_manual_retry(
            request_id,
            PipelineIngressMessage(
                kind="rescan",
                retry_failed=True,
                request_id=request_id,
            ),
        )
        if result is not ManualRetryPublishResult.ACCEPTED:
            raise ValueError("Could not publish the manual retry request")


def resume_prepared(args, current):
    """Keep original document IDs; permit audited implementation-only repairs."""
    original = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
    fields = set(original) | set(current[0])
    for key in fields - {"implementation", "plan_hash"}:
        if original.get(key) != current[0].get(key):
            raise ValueError(f"Resume cannot change frozen plan field: {key}")
    selected = json.loads(
        (args.output / "selected_children.json").read_text(encoding="utf-8")
    )
    if selected != current[1]:
        raise ValueError("Resume requires the exact saved child selection")
    args.resume_audit = {
        "original_plan_hash": original["plan_hash"],
        "original_implementation": original["implementation"],
        "current_implementation": current[0]["implementation"],
        "record_limit_policy": "quarantine surplus rows; preserve 12/16 admission limits",
    }
    return (original, *current[1:])


async def invalidate_malformed_cached_responses(rag, doc_ids, audit_path):
    """Archive and evict only attributable, syntactically invalid extraction JSON."""
    invalid = {}
    for doc_id in doc_ids.values():
        status = await rag.doc_status.get_by_id(doc_id)
        if not status or status.get("status") == DocStatus.PROCESSED.value:
            continue
        keys = status.get("chunks_list") or []
        chunks = await rag.text_chunks.get_by_ids(keys)
        for key, chunk in zip(keys, chunks, strict=True):
            if not chunk:
                continue
            if chunk.get("full_doc_id") != doc_id:
                raise ValueError(
                    "Cached response recovery found a different chunk owner"
                )
            cache_ids = chunk.get("llm_cache_list") or []
            rows = await rag.llm_response_cache.get_by_ids(cache_ids)
            for cache_id, row in zip(cache_ids, rows, strict=True):
                if not row or row.get("cache_type") != "extract":
                    continue
                result = row.get("return")
                if not isinstance(result, str):
                    continue
                try:
                    json.loads(result)
                except json.JSONDecodeError as exc:
                    invalid[cache_id] = {
                        "chunk_id": key,
                        "error": str(exc),
                        "original_cache_row": row,
                    }
    if not invalid:
        return []
    write_json(
        audit_path,
        {
            "reason": "Invalid JSON must be requested again with JSON output mode",
            "entries": invalid,
        },
    )
    await rag.llm_response_cache.delete(list(invalid))
    # Keep native reference-before-cache commit ordering. Chunk cache references
    # remain in place; a missing row is a normal cache miss on explicit retry.
    await rag._flush_storages([rag.text_chunks, rag.llm_response_cache])
    remaining = await rag.llm_response_cache.get_by_ids(list(invalid))
    if any(row is not None for row in remaining):
        raise RuntimeError("Malformed extraction cache eviction did not complete")
    logger.warning(
        "Archived and invalidated %d malformed extraction cache entries; these chunks need new requests, valid cached responses are retained",
        len(invalid),
    )
    return list(invalid)


async def request_extraction(
    settings, messages, response_format, *, output, stem, counters
):
    """Retry transient requests twice; never cache a request failure as a response."""
    attempts = len(EXTRACTION_RETRY_DELAYS) + 1
    for attempt in range(1, attempts + 1):
        counters["extraction_api_attempts"] += 1
        try:
            return await asyncio.to_thread(
                call_model, settings, messages, response_format=response_format
            )
        except ExtractionRequestError as exc:
            write_json(
                output / f"{stem}_attempt_{attempt:02d}_error.json",
                {
                    "error": str(exc),
                    "status": exc.status,
                    "retryable": exc.retryable,
                    "attempt": attempt,
                    "max_attempts": attempts,
                },
            )
            if not exc.retryable:
                raise
            if attempt == attempts:
                raise ExtractionRetriesExhausted(exc, attempts) from exc
            delay = EXTRACTION_RETRY_DELAYS[attempt - 1]
            logger.warning(
                "Extraction %s attempt %d/%d failed: %s; retrying in %ds",
                stem,
                attempt,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


async def run_ingestion(args, prepared):
    plan, selected, indexes, tokenizer, extraction, embedding, values = prepared
    extract_concurrency = getattr(args, "extract_concurrency", 1)
    if not 1 <= extract_concurrency <= 16:
        raise ValueError("Extraction concurrency must be between 1 and 16")
    resume = getattr(args, "resume_failed", False)
    prefix = getattr(args, "receipt_prefix", "")
    # NetworkX is single-writer. This catches the common accidental live server;
    # the operator must also stop other ingestion scripts for this workspace.
    try:
        connection = socket.create_connection(
            ("127.0.0.1", int(values.get("PORT") or 9621)), timeout=1
        )
    except OSError:
        pass
    else:
        connection.close()
        raise ValueError(
            "Stop the running LightRAG backend before writing this same knowledge base"
        )
    if not extraction["api_key"] or not embedding["api_key"]:
        raise ValueError("Extraction and embedding API keys must be configured")
    execution = {
        "extract_concurrency": extract_concurrency,
        "extraction_max_attempts_per_child": len(EXTRACTION_RETRY_DELAYS) + 1,
        "extraction_retry_delays_seconds": list(EXTRACTION_RETRY_DELAYS),
        "max_parallel_documents": 1,
        "embedding_concurrency": 1,
        "embedding_batch_size": 10,
        "embedding_max_attempts_per_batch": len(EMBEDDING_RETRY_DELAYS) + 1,
        "embedding_request_timeout_seconds": EMBEDDING_REQUEST_TIMEOUT,
        "embedding_retry_delays_seconds": list(EMBEDDING_RETRY_DELAYS),
    }
    write_json(args.output / f"{prefix}execution.json", execution)
    logger.info(
        "Ingestion execution: up to %d concurrent extraction requests, one saved child per request; one document writer",
        extract_concurrency,
    )
    counters = {
        "extraction_requests": 0,
        "extraction_api_attempts": 0,
        "embedding_requests": 0,
        "embedding_inputs": 0,
    }
    doc_ids, names = document_identities(plan, selected)
    progress = IngestionProgress(
        plan, selected, doc_ids, args.output / f"{prefix}progress.json", counters
    )

    async def disabled(*args, **kwargs):
        raise ValueError(
            "Answer, keyword, VLM and generic summary calls are disabled in this runner"
        )

    async def extract(prompt, system_prompt=None, **kwargs):
        if (system_prompt or "").strip() != printer.system_prompt().strip():
            raise ValueError("Unexpected non-business model call refused")
        if counters["extraction_requests"] >= plan["selected_child_count"]:
            raise ValueError("Selected-child extraction request budget exhausted")
        counters["extraction_requests"] += 1
        request_number = counters["extraction_requests"]
        response_format = kwargs.get("response_format")
        if response_format != {"type": "json_object"}:
            raise ValueError("Business extraction requires JSON object output mode")
        write_json(
            args.output / f"{prefix}request_{request_number:02d}.json",
            {
                "user_prompt": prompt,
                "system_prompt_sha256": fingerprint(system_prompt),
                "response_format": response_format,
            },
        )
        response = await request_extraction(
            extraction,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            output=args.output,
            stem=f"{prefix}request_{request_number:02d}",
            counters=counters,
        )
        write_json(
            args.output / f"{prefix}response_{request_number:02d}.json",
            response,
        )
        progress.last_response = f"{prefix}response_{request_number:02d}.json"
        progress.save()
        choice = response["choices"][0]
        text = choice["message"]["content"]
        return (
            TruncatedResponse(text) if choice.get("finish_reason") == "length" else text
        )

    async def embed(texts):
        progress.stage = "embedding"
        counters["embedding_requests"] += 1
        counters["embedding_inputs"] += len(texts)
        request_number = counters["embedding_requests"]
        vectors, usage = await asyncio.to_thread(embed_batch, embedding, texts)
        write_json(
            args.output / f"{prefix}embedding_usage_{request_number:03d}.json",
            usage,
        )
        progress.save()
        logger.info(
            "Vector indexing: embedding batch %d returned %d vectors; extraction %d/%d chunks; ingestion not yet confirmed",
            request_number,
            len(texts),
            sum(progress.completed.values()),
            sum(progress.expected.values()),
        )
        return vectors

    full_ingestion = plan.get("selection_mode") == "all_saved_children"
    groups = {value: key for key, value in doc_ids.items()}
    rag = None

    @accepts_chunking_context
    async def existing_children(_tokenizer, content, *args_, context):
        group = groups[context.doc_id]
        progress.start_document(group)
        chunks, bundle = prepared_chunks(
            selected[group],
            indexes[group],
            context.doc_id,
            names[group],
            tokenizer,
            include_all_parents=full_ingestion,
        )
        record = await rag.full_docs.get_by_id(context.doc_id)
        await rag.full_docs.upsert({context.doc_id: {**record, "parent_child": bundle}})
        return chunks

    load_dotenv(args.env_file, override=False)
    rag = LightRAG(
        working_dir=plan["working_dir"],
        workspace=plan["workspace"],
        **plan["stores"],
        tokenizer=tokenizer,
        chunking_func=existing_children,
        addon_params={"extraction_profile": "printer_v2"},
        llm_model_func=disabled,
        role_llm_configs={
            "extract": RoleLLMConfig(
                func=extract,
                max_async=extract_concurrency,
                timeout=(
                    extraction["timeout"] * (len(EXTRACTION_RETRY_DELAYS) + 1)
                    + sum(EXTRACTION_RETRY_DELAYS)
                    + 30
                ),
                metadata={
                    "binding": "openai",
                    "model": extraction["model"],
                    "host": extraction["base_url"],
                    "provider_options": extraction["extra_body"],
                },
            )
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding["dimension"],
            max_token_size=embedding["token_limit"],
            model_name=embedding["model"],
            func=embed,
        ),
        max_parallel_insert=1,
        llm_model_max_async=extract_concurrency,
        embedding_func_max_async=1,
        default_embedding_timeout=(
            EMBEDDING_REQUEST_TIMEOUT * (len(EMBEDDING_RETRY_DELAYS) + 1)
            + sum(EMBEDDING_RETRY_DELAYS)
            + 30
        ),
        embedding_batch_num=10,
        entity_extract_max_gleaning=0,
        force_llm_summary_on_merge=1000,
        summary_max_tokens=8192,
    )
    await rag.initialize_storages()
    logger.addHandler(progress)
    try:
        queued = await rag.doc_status.get_docs_by_statuses(
            [
                status
                for status in DocStatus
                if status not in {DocStatus.PROCESSED, DocStatus.FAILED}
            ],
            strict=True,
        )
        if queued and not (
            resume and full_ingestion and set(queued) <= set(doc_ids.values())
        ):
            raise ValueError(
                "Existing documents are queued or unfinished; resolve them before this bounded sample run"
            )
        existing = await rag.doc_status.get_by_ids(list(doc_ids.values()))
        if any(existing) and not resume:
            raise ValueError(
                "This sample run already has document records; inspect them before retrying"
            )
        progress.seed_processed(doc_ids, existing)
        for (group, _), row in zip(doc_ids.items(), existing, strict=True):
            if row and row.get("status") == DocStatus.PROCESSED.value:
                saved_keys = row.get("chunks_list") or []
                saved_chunks = await rag.text_chunks.get_by_ids(saved_keys)
                progress.skipped[group] = {
                    key
                    for key, chunk in zip(saved_keys, saved_chunks, strict=True)
                    if (chunk or {})
                    .get("business_extraction", {})
                    .get("review", {})
                    .get("skipped")
                }
        progress.save()
        invalidated_cache_entries = []
        if resume:
            await publish_sample_retry(rag, doc_ids, names, plan)
        else:
            await rag.apipeline_enqueue_documents(
                [
                    "\n\n".join(child_evidence_content(c) for c in selected[g])
                    for g in selected
                ],
                ids=list(doc_ids.values()),
                file_paths=list(names.values()),
                process_options="C",
            )
        for doc_id in doc_ids.values():
            if await rag.full_docs.get_by_id(doc_id) is None:
                raise ValueError(
                    "A sample was deduplicated; no full-document processing was started"
                )
        await rag.apipeline_process_enqueue_documents()
        report = {
            "plan_hash": plan["plan_hash"],
            "documents": {},
            "calls": counters,
            "execution": execution,
            "semantic_review_required": True,
            "resume_failed": resume,
            "invalidated_cache_entries": invalidated_cache_entries,
        }
        skipped_details = []
        for group, doc_id in doc_ids.items():
            status = await rag.doc_status.get_by_id(doc_id)
            document = await rag.full_docs.get_by_id(doc_id)
            keys = (status or {}).get("chunks_list", [])
            chunks = await rag.text_chunks.get_by_ids(keys)
            skipped_for_document = []
            for key, chunk in zip(keys, chunks, strict=True):
                review = (chunk or {}).get("business_extraction", {}).get("review", {})
                if review.get("skipped"):
                    skipped_for_document.append(key)
                    skipped_details.append(
                        {
                            "doc_id": doc_id,
                            "chunk_id": key,
                            "parent_id": chunk.get("parent_child", {}).get("parent_id"),
                            "heading_path": chunk.get("parent_child", {}).get(
                                "heading_path", []
                            ),
                            "error": review["sample_error"],
                        }
                    )
            progress.skipped[group] = set(skipped_for_document)
            write_json(
                args.output / f"{prefix}stored_chunks_{group}.json",
                dict(zip(keys, chunks, strict=True)),
            )
            anchor = await rag.full_entities.get_by_id(doc_id)
            names_in_graph = (anchor or {}).get("entity_names", [])
            nodes = {
                name: await rag.chunk_entity_relation_graph.get_node(name)
                for name in names_in_graph
            }
            write_json(args.output / f"{prefix}graph_nodes_{group}.json", nodes)
            write_json(
                args.output / f"{prefix}relation_anchor_{group}.json",
                await rag.full_relations.get_by_id(doc_id),
            )
            relation_anchor = await rag.full_relations.get_by_id(doc_id)
            pairs = (relation_anchor or {}).get("relation_pairs", [])
            write_json(
                args.output / f"{prefix}graph_edges_{group}.json",
                [
                    {
                        "source": pair[0],
                        "target": pair[1],
                        "stored_edge": await rag.chunk_entity_relation_graph.get_edge(
                            *pair
                        ),
                    }
                    for pair in pairs
                ],
            )
            saved = {
                (c or {}).get("parent_child", {}).get("child_id"): c for c in chunks
            }
            intact = len(chunks) == len(selected[group]) and all(
                saved.get(c["child_id"], {}).get("content") == child_evidence_content(c)
                for c in selected[group]
            )
            report["documents"][group] = {
                "doc_id": doc_id,
                "file_path": names[group],
                "status": (status or {}).get("status"),
                "error": (status or {}).get("error_msg"),
                "selected_children": len(selected[group]),
                "stored_children": len(keys),
                "intact_saved_children": intact,
                "skipped_extraction_chunks": len(skipped_for_document),
                "quarantined_records": sum(
                    len(
                        (c or {})
                        .get("business_extraction", {})
                        .get("review", {})
                        .get("rejected", [])
                    )
                    for c in chunks
                ),
            }
            if full_ingestion:
                report["documents"][group]["stored_parents"] = len(
                    (document or {}).get("parent_child", {}).get("parents", {})
                )
        successful = all(
            d["status"] == DocStatus.PROCESSED.value
            and d["intact_saved_children"]
            and (not full_ingestion or d["stored_parents"] == len(indexes[g].parents))
            for g, d in report["documents"].items()
        )
        report["skipped_extraction_chunks"] = len(skipped_details)
        report["completed_with_skips"] = successful and bool(skipped_details)
        write_json(args.output / f"{prefix}skipped_chunks.json", skipped_details)
        # Persist verification before optional sample cleanup. A cleanup failure
        # must never conceal that the full document writes already succeeded.
        write_json(args.output / f"{prefix}summary.json", report)
        if successful and full_ingestion and not skipped_details:
            try:
                report["replaced_samples"] = await remove_replaced_samples(
                    rag, plan, selected
                )
            except Exception as exc:
                report["sample_cleanup_error"] = str(exc)
                write_json(args.output / f"{prefix}summary.json", report)
                raise
            write_json(args.output / f"{prefix}summary.json", report)
        elif successful and full_ingestion and plan.get("replace_sample_plan_hash"):
            report["sample_cleanup_deferred"] = (
                "Skipped extraction chunks require review before deleting existing samples"
            )
            write_json(args.output / f"{prefix}summary.json", report)
        print(json.dumps(report, ensure_ascii=False))
        progress.stage = (
            "completed_with_skips"
            if report["completed_with_skips"]
            else ("completed" if successful else "failed")
        )
        progress.save()
        logger.info(
            "Ingestion %s: reviewed %d/%d chunks; skipped %d chunks; processed documents %d/%d",
            progress.stage,
            sum(progress.completed.values()),
            sum(progress.expected.values()),
            len(skipped_details),
            sum(
                d["status"] == DocStatus.PROCESSED.value
                for d in report["documents"].values()
            ),
            len(report["documents"]),
        )
        if not successful:
            # Preserve raw source/cache attribution while discarding regenerable
            # failed vector buffers before finalize shuts down provider queues.
            await rag._discard_pending_index_ops()
        return 0 if successful else 2
    finally:
        logger.removeHandler(progress)
        progress.close()
        await rag.finalize_storages()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--all-children", action="store_true")
    parser.add_argument("--replace-sample-run", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--bundles", type=Path, default=Path("outputs/printer_parent_child/20261017_v3")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/printer_child_ingestion/saved_children_v1"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tmp/parent_blocks_tools/cl100k_base.tiktoken"),
    )
    parser.add_argument("--per-document", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--max-output-tokens", type=int, default=6144)
    parser.add_argument(
        "--extract-concurrency", type=int, choices=range(1, 17), default=1
    )
    args = parser.parse_args(argv)
    if args.replace_sample_run is not None and not args.all_children:
        parser.error("--replace-sample-run requires --all-children")
    if args.all_children and args.output == Path(
        "outputs/printer_child_ingestion/saved_children_v1"
    ):
        parser.error("Full ingestion requires its own --output receipt directory")
    prepared = prepare(args)
    if args.resume_failed:
        prepared = resume_prepared(args, prepared)
    plan, selected = prepared[:2]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Plan changed; use a new output directory")
    if not args.resume_failed:
        write_json(path, plan)
        write_json(args.output / "selected_children.json", selected)
    print(
        json.dumps(
            {
                "mode": "run" if args.run else "preflight",
                "resume_failed": args.resume_failed,
                "working_dir": plan["working_dir"],
                "workspace": plan["workspace"],
                "children": plan["selected_child_count"],
                "extraction_requests_at_most": plan["selected_child_count"],
                "extraction_api_attempts_at_most": (
                    plan["selected_child_count"] * (len(EXTRACTION_RETRY_DELAYS) + 1)
                ),
                "extract_concurrency": args.extract_concurrency,
                "embedding": "entities, relations and selected children (run only)",
                "parsing_requests": 0,
                "answer_requests": 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not args.run:
        return 0
    with experiment_lock(args.output):
        if args.resume_failed:
            args.receipt_prefix = (
                datetime.now(timezone.utc).strftime("resume_%Y%m%dT%H%M%S_")
                + uuid4().hex[:8]
                + "_"
            )
            write_json(
                args.output / f"{args.receipt_prefix}audit.json", args.resume_audit
            )
        return asyncio.run(run_ingestion(args, prepared))


if __name__ == "__main__":
    raise SystemExit(main())
