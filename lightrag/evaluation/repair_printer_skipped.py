"""Repair at most ten explicitly listed skipped chunks in the existing index.

Default is offline preflight. --run requires the backend to be stopped and calls
extraction/embedding services. See docs/PrinterSkippedExtractionRepair-zh.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from dotenv import dotenv_values, load_dotenv

from lightrag import LightRAG
from lightrag.chunk_schema import format_heading_context
from lightrag.document_structure.children import load_local_tokenizer
from lightrag.evaluation.batch_baseline import write_json
from lightrag.evaluation.compare_parent_child_retrieval import (
    EMBEDDING_REQUEST_TIMEOUT,
    EMBEDDING_RETRY_DELAYS,
    embed_batch,
    embedding_config,
    experiment_lock,
    provider_identity,
)
from lightrag.evaluation.ingest_printer_children import (
    STORES,
    request_extraction,
)
from lightrag.evaluation.printer_graph_smoke import config, public_config
from lightrag.extraction import printer
from lightrag.extraction.errors import ExtractionRetriesExhausted
from lightrag.extraction.runtime import (
    build_records,
    chunk_context,
    fingerprint,
    snapshot_config,
)
from lightrag.operate import collect_kg_merge_candidates, merge_nodes_and_edges
from lightrag.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    logger,
    make_relation_chunk_key,
    sanitize_text_for_encoding,
)

VERSION = "printer-skipped-extraction-repair-v1"
SOURCE_KEYS = (
    "content",
    "full_doc_id",
    "file_path",
    "parent_child",
    "heading",
    "extraction_config",
    "tokens",
    "chunk_order_index",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_hash(chunk):
    return fingerprint({key: chunk.get(key) for key in SOURCE_KEYS})


def completed(chunk, plan_hash):
    marker = chunk.get("business_extraction_repair", {})
    return marker.get("plan_hash") == plan_hash and marker.get("phase") == "complete"


def require_stopped(values):
    port = int(values.get("PORT") or 9621)
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=1)
    except OSError:
        return
    connection.close()
    raise ValueError("Stop the LightRAG backend and other writers before --run")


def prepare(args):
    values = {**dotenv_values(args.env_file), **os.environ}
    directory = Path(values.get("WORKING_DIR") or "rag_storage").resolve()
    workspace = str(values.get("WORKSPACE") or "")
    storage = directory / workspace
    stores = {key: values.get(env) or default for key, (env, default) in STORES.items()}
    if any(stores[key] != default for key, (_, default) in STORES.items()):
        raise ValueError(
            "This bounded repair requires the existing JSON/NetworkX/Nano stores"
        )
    listed = read_json(args.skipped_file)
    ids = [row["chunk_id"] for row in listed]
    if not 1 <= len(ids) <= 10 or len(set(ids)) != len(ids):
        raise ValueError("Supply one to ten distinct explicitly skipped source chunks")
    chunks = read_json(storage / "kv_store_text_chunks.json")
    cache = read_json(storage / "kv_store_llm_response_cache.json")
    statuses = read_json(storage / "kv_store_doc_status.json")
    docs = read_json(storage / "kv_store_full_docs.json")
    saved_plan = (
        read_json(args.output / "plan.json")
        if (args.output / "plan.json").exists()
        else None
    )
    selected, original_cache = {}, {}
    for source in listed:
        key = source["chunk_id"]
        chunk = chunks[key]
        doc = chunk["full_doc_id"]
        if doc != source["doc_id"] or statuses[doc]["status"] != "processed":
            raise ValueError(
                "Listed source ownership or processed document state changed"
            )
        if key not in statuses[doc].get("chunks_list", []):
            raise ValueError("A target chunk is absent from its document's chunk list")
        parent = chunk.get("parent_child", {}).get("parent_id")
        if (
            parent != source["parent_id"]
            or parent not in docs[doc]["parent_child"]["parents"]
        ):
            raise ValueError(
                "A target's persisted parent mapping is missing or changed"
            )
        if chunk.get("extraction_config") != snapshot_config("printer_v2"):
            raise ValueError("A target uses a different extraction contract")
        evidence = chunk.get("business_extraction", {})
        marker = chunk.get("business_extraction_repair", {})
        is_ours = (
            saved_plan
            and marker.get("plan_hash") == saved_plan["plan_hash"]
            and marker.get("phase") in {"complete", "pending_cache"}
        )
        if not is_ours and (
            not evidence.get("review", {}).get("skipped")
            or evidence.get("nodes")
            or evidence.get("facts")
        ):
            raise ValueError(
                "Only skipped chunks with no accepted contributions can be repaired"
            )
        refs = chunk.get("llm_cache_list") or []
        if len(refs) != 1:
            raise ValueError(
                "Expected one attributable malformed extraction cache row per target"
            )
        cache_key = refs[0]
        entry = cache.get(cache_key)
        owners = [
            k for k, c in chunks.items() if cache_key in (c.get("llm_cache_list") or [])
        ]
        if (
            owners != [key]
            or not entry
            or entry.get("cache_type") != "extract"
            or entry.get("chunk_id") != key
        ):
            raise ValueError(
                "Repair refuses shared, missing or unattributable cache rows"
            )
        selected[key] = chunk
        original_cache[cache_key] = entry
    extraction = config(args.env_file, args.max_output_tokens)
    embedding = embedding_config(args.env_file)
    plan = {
        "version": VERSION,
        "working_dir": str(directory),
        "workspace": workspace,
        "stores": stores,
        "contract": snapshot_config("printer_v2"),
        "extract": public_config(extraction),
        "embedding": provider_identity(embedding),
        "sources": {k: source_hash(c) for k, c in selected.items()},
        "cache_prompts": {
            k: fingerprint(c.get("original_prompt")) for k, c in original_cache.items()
        },
    }
    plan["plan_hash"] = fingerprint(plan)
    if saved_plan and saved_plan != plan:
        raise ValueError(
            "Frozen repair plan changed; do not reuse this output directory"
        )
    return plan, selected, original_cache, extraction, embedding, values


def records(chunk_id, chunk, response, timestamp):
    context = chunk_context(
        chunk, chunk_id, chunk["extraction_config"], format_heading_context(chunk)
    )
    choice = response["choices"][0]
    return build_records(
        sanitize_text_for_encoding(chunk["content"], strip=False),
        context,
        choice["message"]["content"],
        timestamp,
        on_schema_error="skip",
        response_truncated=choice.get("finish_reason") != "stop",
    )


def extraction_messages(key, chunk):
    context = chunk_context(
        chunk, key, chunk["extraction_config"], format_heading_context(chunk)
    )
    return printer.messages(
        {
            "document_name": context["document_name"],
            "heading_path": context["heading_path"],
            "domain": context["domain"],
            "input_text": sanitize_text_for_encoding(chunk["content"], strip=False),
        }
    )


def repair_mixed_quotes(response):
    """Escape mixed curly-open/ASCII-close quotes inside completed JSON strings.

    Only backslashes are inserted. Do not complete truncated output, invent
    fields or relax the business validator. Return an auditable derived copy.
    """
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Cannot repair a truncated or missing saved completion")
    text = choices[0]["message"]["content"]
    try:
        json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        raise ValueError("Saved response has no mixed-quote JSON syntax error")
    result, changes = [], []
    inside, escaped, curly_depth = False, False, 0
    for offset, char in enumerate(text):
        if inside:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "“":
                curly_depth += 1
            elif char == "”":
                curly_depth = max(0, curly_depth - 1)
            elif char == '"':
                following = text[offset + 1 :].lstrip()
                if curly_depth and following and following[0] not in ",]}:":
                    result.append("\\")
                    changes.append({"offset": offset, "insert": "\\", "before": '"'})
                    curly_depth -= 1
                else:
                    inside, curly_depth = False, 0
        elif char == '"':
            inside, curly_depth = True, 0
        result.append(char)
    repaired = "".join(result)
    if not changes:
        raise ValueError("No narrowly repairable mixed quotes found in saved response")
    json.loads(repaired)
    derived = deepcopy(response)
    derived["choices"][0]["message"]["content"] = repaired
    return derived, changes


async def extract_targets(args, plan, selected, settings, counters):
    valid, failures = {}, {}
    for number, (key, chunk) in enumerate(selected.items(), 1):
        path = args.output / f"response_{key}.json"
        prior = read_json(path) if path.exists() else None
        if completed(chunk, plan["plan_hash"]):
            continue
        if getattr(args, "repair_saved_quotes", False) and (
            prior is None or prior.get("validation_failed")
        ):
            try:
                if prior is None or "response" not in prior:
                    raise ValueError(
                        "No completed saved response to recover; no model call made"
                    )
                derived, changes = repair_mixed_quotes(prior["response"])
                candidate = records(key, chunk, derived, prior["timestamp"])
                if candidate[2]["review"].get("skipped"):
                    raise ValueError(candidate[2]["review"]["sample_error"])
            except (ValueError, KeyError, TypeError) as exc:
                failures[key] = str(exc)
                continue
            audit_name = f"syntax_repair_{key}_attempt_{prior['attempt']}.json"
            audit = {
                "policy": "escape_mixed_curly_ascii_quotes_v1",
                "plan_hash": plan["plan_hash"],
                "chunk_id": key,
                "original_response": prior["response"],
                "derived_response": derived,
                "changes": changes,
            }
            write_json(args.output / audit_name, audit)
            prior = {
                **prior,
                "response": derived,
                "validation_failed": False,
                "syntax_repair": {
                    "audit_file": audit_name,
                    "inserted_backslashes": len(changes),
                    "original_response_sha256": fingerprint(audit["original_response"]),
                },
            }
            write_json(path, prior)
            logger.info(
                "Recovered saved response for %s: %d quote escapes, no model request",
                key,
                len(changes),
            )
        if prior is None or prior.get("validation_failed"):
            if prior and not args.resume:
                raise ValueError(
                    "Use --resume to explicitly retry an unusable saved response"
                )
            attempts = prior.get("attempt", 0) + 1 if prior else 1
            messages = extraction_messages(key, chunk)
            stem = f"{key}_attempt_{attempts}"
            response_format = (
                None
                if getattr(args, "plain_json_retry", False)
                else {"type": "json_object"}
            )
            write_json(
                args.output / f"request_{stem}.json",
                {
                    "plan_hash": plan["plan_hash"],
                    "chunk_id": key,
                    "messages": messages,
                    "response_format": response_format,
                },
            )
            counters["extraction_requests"] += 1
            try:
                response = await request_extraction(
                    settings,
                    messages,
                    response_format,
                    output=args.output,
                    stem=stem,
                    counters=counters,
                )
            except ExtractionRetriesExhausted as exc:
                failures[key] = str(exc)
                write_json(
                    path,
                    {"attempt": attempts, "validation_failed": True, "error": str(exc)},
                )
                continue
            prior = {
                "response": response,
                "timestamp": int(time.time()),
                "attempt": attempts,
            }
            # Immutable attempt receipts precede the reusable per-source pointer.
            write_json(args.output / f"response_{stem}.json", prior)
            write_json(path, prior)
        result = records(key, chunk, prior["response"], prior["timestamp"])
        evidence = result[2]
        write_json(args.output / f"review_{key}.json", evidence)
        if evidence["review"].get("skipped"):
            failures[key] = evidence["review"]["sample_error"]
            write_json(path, {**prior, "validation_failed": True})
        else:
            valid[key] = prior
        logger.info(
            "Repair extraction %d/%d: %d entities, %d relationships, usable=%s",
            number,
            len(selected),
            len(result[0]),
            len(result[1]),
            key in valid,
        )
    return valid, failures


async def verify_graph(rag, results):
    for nodes, edges, _ in results.values():
        for name, records_ in nodes.items():
            tracked = await rag.entity_chunks.get_by_id(name) or {}
            if not {r["source_id"] for r in records_} <= set(
                tracked.get("chunk_ids", [])
            ):
                raise RuntimeError("Repaired entity lost its source attribution")
            if not await rag.chunk_entity_relation_graph.has_node(name):
                raise RuntimeError("Repaired entity missing from graph")
            if not await rag.entities_vdb.get_by_id(
                compute_mdhash_id(name, prefix="ent-")
            ):
                raise RuntimeError("Repaired entity missing from vector index")
        for pair, records_ in edges.items():
            left, right = sorted(pair)
            tracked = (
                await rag.relation_chunks.get_by_id(
                    make_relation_chunk_key(left, right)
                )
                or {}
            )
            if not {r["source_id"] for r in records_} <= set(
                tracked.get("chunk_ids", [])
            ):
                raise RuntimeError("Repaired relation lost its source attribution")
            if not await rag.chunk_entity_relation_graph.has_edge(left, right):
                raise RuntimeError("Repaired relation missing from graph")
            if not await rag.relationships_vdb.get_by_id(
                compute_mdhash_id(left + right, prefix="rel-")
            ):
                raise RuntimeError("Repaired relation missing from vector index")


async def apply_document(rag, doc_id, selected, valid, plan, output):
    """Prewrite UNION anchors, merge, then publish repaired evidence/cache.

    A partial merge retains conservative base anchors and the original skipped
    evidence until retry completes. See the repair guide for recovery residues.
    """
    results = {}
    for key, attempt in valid.items():
        current = await rag.text_chunks.get_by_id(key)
        if current is None or source_hash(current) != plan["sources"][key]:
            raise ValueError("Source changed after repair preflight")
        if completed(current, plan["plan_hash"]):
            continue
        results[key] = records(key, current, attempt["response"], attempt["timestamp"])
    if not results:
        return []
    if not await rag.full_entities.get_by_id(
        doc_id
    ) or not await rag.full_relations.get_by_id(doc_id):
        raise ValueError(
            "Original document recovery anchors must exist before additive repair"
        )
    candidates, pairs = collect_kg_merge_candidates(
        [(a, b) for a, b, _ in results.values()]
    )
    journal = {
        "plan_hash": plan["plan_hash"],
        "doc_id": doc_id,
        "chunks": list(results),
        "entity_names": sorted(candidates),
        "relation_pairs": sorted(pairs),
        "phase": "prepared",
        "response_sha256": {
            key: fingerprint(valid[key]["response"]) for key in results
        },
    }
    journal_path = output / f"apply_{doc_id}.json"
    previous = read_json(journal_path) if journal_path.exists() else {}
    graph_committed = (
        previous.get("plan_hash") == plan["plan_hash"]
        and previous.get("phase") in {"graph_committed", "complete"}
        and all(
            previous.get("response_sha256", {}).get(key) == value
            for key, value in journal["response_sha256"].items()
        )
    )
    # Phase 0 in merge_nodes_and_edges REPLACES whole-document anchors. A subset
    # repair must instead prewrite a union, then disable that replace-only phase.
    if not graph_committed:
        write_json(journal_path, journal)
        await rag._union_doc_recovery_anchors(doc_id, candidates, pairs)
        await merge_nodes_and_edges(
            chunk_results=[(a, b) for a, b, _ in results.values()],
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entity_vdb=rag.entities_vdb,
            relationships_vdb=rag.relationships_vdb,
            global_config=rag._build_global_config(),
            doc_id=doc_id,
            full_entities_storage=None,
            full_relations_storage=None,
            pipeline_status={"history_messages": [], "latest_message": ""},
            pipeline_status_lock=asyncio.Lock(),
            llm_response_cache=rag.llm_response_cache,
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
            file_path=selected[next(iter(results))]["file_path"],
        )
        await rag._insert_done()
    await verify_graph(rag, results)
    write_json(output / f"apply_{doc_id}.json", {**journal, "phase": "graph_committed"})
    cache_updates = {}
    for key, (_, _, evidence) in results.items():
        attempt = valid[key]
        current = await rag.text_chunks.get_by_id(key)
        if current is None or source_hash(current) != plan["sources"][key]:
            raise ValueError("Source changed before evidence publication")
        cache_key = current["llm_cache_list"][0]
        cache_row = await rag.llm_response_cache.get_by_id(cache_key)
        if (
            not cache_row
            or cache_row.get("chunk_id") != key
            or fingerprint(cache_row.get("original_prompt"))
            != plan["cache_prompts"][cache_key]
        ):
            raise ValueError("Extraction cache attribution changed before publication")
        text = attempt["response"]["choices"][0]["message"]["content"]
        cache_updates[cache_key] = {**cache_row, "return": text}
        await rag.text_chunks.upsert(
            {
                key: {
                    **current,
                    "business_extraction": evidence,
                    "business_extraction_attempt": {
                        "context": evidence["context"],
                        "raw_response": text,
                        "timestamp": attempt["timestamp"],
                        "truncated": False,
                    },
                    "business_extraction_repair": {
                        "plan_hash": plan["plan_hash"],
                        "phase": "pending_cache",
                        "response_sha256": fingerprint(attempt["response"]),
                        **(
                            {"syntax_repair": attempt["syntax_repair"]}
                            if attempt.get("syntax_repair")
                            else {}
                        ),
                    },
                }
            }
        )
    # Source attribution stays durable BEFORE replacement cache rows, in the
    # same reference-before-cache direction as normal extraction.
    await rag._flush_storages([rag.text_chunks])
    await rag.llm_response_cache.upsert(cache_updates)
    await rag._flush_storages([rag.llm_response_cache])
    for key in results:
        current = await rag.text_chunks.get_by_id(key)
        marker = {**current["business_extraction_repair"], "phase": "complete"}
        await rag.text_chunks.upsert(
            {key: {**current, "business_extraction_repair": marker}}
        )
    await rag._flush_storages([rag.text_chunks])
    write_json(output / f"apply_{doc_id}.json", {**journal, "phase": "complete"})
    return list(results)


async def run(args, prepared):
    plan, selected, _, extraction, embedding, values = prepared
    require_stopped(values)
    if not extraction["api_key"] or not embedding["api_key"]:
        raise ValueError("Extraction and embedding keys must be configured")
    archive = args.output / "workspace_before.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.tmp")
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as backup:
            for path in (Path(plan["working_dir"]) / plan["workspace"]).glob("*"):
                if path.is_file():
                    backup.write(path, path.name)
        temporary.replace(archive)
    with ZipFile(archive) as backup:
        if backup.testzip() is not None:
            raise ValueError("Workspace backup failed its integrity check")
    counters = {
        "extraction_requests": 0,
        "extraction_api_attempts": 0,
        "embedding_requests": 0,
    }
    valid, failures = await extract_targets(args, plan, selected, extraction, counters)
    require_stopped(values)

    async def disabled(*_args, **_kwargs):
        raise ValueError(
            "Additional extraction/summary/answer calls are disabled during repair merge"
        )

    async def embed(texts):
        counters["embedding_requests"] += 1
        vectors, usage = await asyncio.to_thread(embed_batch, embedding, texts)
        write_json(args.output / f"embedding_usage_{time.time_ns()}.json", usage)
        return vectors

    load_dotenv(args.env_file, override=False)
    rag = LightRAG(
        working_dir=plan["working_dir"],
        workspace=plan["workspace"],
        **plan["stores"],
        tokenizer=load_local_tokenizer(args.tokenizer),
        addon_params={"extraction_profile": "printer_v2"},
        llm_model_func=disabled,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding["dimension"],
            max_token_size=embedding["token_limit"],
            model_name=embedding["model"],
            func=embed,
        ),
        embedding_func_max_async=1,
        embedding_batch_num=10,
        default_embedding_timeout=EMBEDDING_REQUEST_TIMEOUT
        * (len(EMBEDDING_RETRY_DELAYS) + 1)
        + 60,
        force_llm_summary_on_merge=1000,
        summary_max_tokens=8192,
    )
    await rag.initialize_storages()
    try:
        async with rag._admin_write_gate("repair_printer_skipped"):
            # A failed prior call may have published evidence before its cache.
            # Keep that state eligible for explicit roll-forward, never new LLM input.
            grouped = defaultdict(dict)
            for key, attempt in valid.items():
                grouped[selected[key]["full_doc_id"]][key] = attempt
            for doc_id, attempts in grouped.items():
                await apply_document(rag, doc_id, selected, attempts, plan, args.output)
            rows = await rag.text_chunks.get_by_ids(list(selected))
            repaired = [
                k
                for k, c in zip(selected, rows, strict=True)
                if c and completed(c, plan["plan_hash"])
            ]
            remaining = [k for k in selected if k not in repaired]
            details = []
            for key, row in zip(selected, rows, strict=True):
                evidence = (row or {}).get("business_extraction", {})
                nodes, facts = evidence.get("nodes", {}), evidence.get("facts", [])
                details.append(
                    {
                        "chunk_id": key,
                        "repaired": key in repaired,
                        "accepted_entities": len(nodes),
                        "accepted_relationships": len(facts),
                        "quarantined_records": len(
                            evidence.get("review", {}).get("rejected", [])
                        ),
                        "empty_graph": key in repaired and not nodes and not facts,
                    }
                )
            report = {
                "plan_hash": plan["plan_hash"],
                "targeted_chunks": len(selected),
                "repaired_chunks": len(repaired),
                "repaired_chunk_ids": repaired,
                "remaining_skipped_chunks": len(remaining),
                "remaining_chunk_ids": remaining,
                "extraction_failures": failures,
                "calls_this_run": counters,
                "plain_json_retry": bool(getattr(args, "plain_json_retry", False)),
                "repair_saved_quotes": bool(
                    getattr(args, "repair_saved_quotes", False)
                ),
                "accepted_entities": sum(
                    len(c["business_extraction"].get("nodes", {}))
                    for c in rows
                    if c and completed(c, plan["plan_hash"])
                ),
                "accepted_relationships": sum(
                    len(c["business_extraction"].get("facts", []))
                    for c in rows
                    if c and completed(c, plan["plan_hash"])
                ),
                "chunks": details,
                "semantic_review_required": True,
            }
            write_json(args.output / f"summary_{time.time_ns()}.json", report)
            write_json(args.output / "summary.json", report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            return 2 if remaining else 0
    except BaseException:
        # Keep durable base anchors and raw receipts, never restore an old whole
        # namespace or discard source attribution after a partial graph write.
        await rag._discard_pending_index_ops()
        raise
    finally:
        await rag.finalize_storages()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plain-json-retry", action="store_true")
    parser.add_argument("--repair-saved-quotes", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--skipped-file",
        type=Path,
        default=Path(
            "outputs/printer_full_ingestion/saved_children_v1/resume_20261018T072912_b43c2e92_skipped_chunks.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/printer_skipped_repair/saved_children_v1"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tmp/parent_blocks_tools/cl100k_base.tiktoken"),
    )
    parser.add_argument("--max-output-tokens", type=int, default=6144)
    args = parser.parse_args(argv)
    if (args.plain_json_retry or args.repair_saved_quotes) and not args.resume:
        parser.error("Recovery options require an explicit --resume")
    if args.plain_json_retry and args.repair_saved_quotes:
        parser.error("Choose either a model retry or saved-response quote recovery")
    args.output.mkdir(parents=True, exist_ok=True)
    with experiment_lock(args.output):
        prepared = prepare(args)
        plan, selected, cache = prepared[:3]
        if not (args.output / "plan.json").exists():
            write_json(args.output / "selected_chunks_before.json", selected)
            write_json(args.output / "cache_rows_before.json", cache)
            write_json(args.output / "plan.json", plan)
        print(
            json.dumps(
                {
                    "mode": "run" if args.run else "preflight",
                    "workspace": plan["workspace"],
                    "targeted_chunks": len(selected),
                    "already_repaired_chunks": sum(
                        completed(c, plan["plan_hash"]) for c in selected.values()
                    ),
                    "plain_json_retry": args.plain_json_retry,
                    "repair_saved_quotes": args.repair_saved_quotes,
                    "extraction_requests_at_most": 0
                    if args.repair_saved_quotes
                    else sum(
                        not completed(c, plan["plan_hash"]) for c in selected.values()
                    ),
                    "extraction_api_attempts_at_most": 0
                    if args.repair_saved_quotes
                    else 3
                    * sum(
                        not completed(c, plan["plan_hash"]) for c in selected.values()
                    ),
                    "embedding": "only accepted entity/relation indexing; no child-vector rebuild",
                    "parsing_requests": 0,
                    "answer_requests": 0,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return asyncio.run(run(args, prepared)) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
