"""Opt-in business extraction, provenance and native graph identity conversion.

The same conversion is used during initial extraction and cache reconstruction.
See docs/PrinterGraphSystemIntegration-zh.md for identity and lifecycle rules.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from copy import deepcopy

from lightrag.extraction import printer

VERSION = "printer-native-graph-v1"
MODEL_RE = re.compile(r"\b[A-Z]{1,6}\d{3,6}\b")
IDENTITY_MARKER = " ⟦printer:"


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def snapshot_config(profile: str = "generic") -> dict:
    """Freeze the selected contract for new documents; generic is the default."""
    if profile not in {"generic", "printer_v2"}:
        raise ValueError("extraction_profile must be generic or printer_v2")
    if profile == "generic":
        return {"profile": profile}
    return {
        "profile": profile,
        "version": VERSION,
        "contract": printer.VERSION,
        "prompt_sha256": hashlib.sha256(printer.system_prompt().encode()).hexdigest(),
    }


def assert_contract(config: dict) -> None:
    if config != snapshot_config("printer_v2"):
        raise ValueError(
            "Business extraction contract changed; explicitly reprocess with the intended contract"
        )


def chunk_context(chunk: dict, chunk_id: str, config: dict, heading: str) -> dict:
    """Attach real source ownership regardless of the chunking strategy."""
    assert_contract(config)
    doc_id = chunk.get("full_doc_id")
    if not doc_id:
        raise ValueError("Business extraction requires an owning document ID")
    parent = chunk.get("parent_child") or {}
    if parent and parent["doc_id"] != doc_id:
        raise ValueError("Business extraction parent has a different document owner")
    name = chunk.get("file_path", "unknown_source")
    models = sorted(set(MODEL_RE.findall(name.upper().replace("_", " "))))
    scope = deepcopy(
        parent.get("scope")
        or {
            "status": "filename" if models else "unknown",
            "applicable_models": models,
        }
    )
    if scope.get("status") == "conflict":
        raise ValueError(
            "Business extraction requires resolution of the conflicting parent scope"
        )
    heading = heading[:2000]
    domain = parent.get("feature_domain", "unknown")
    if domain == "unknown":
        network = bool(re.search(r"网络(?:连接)?报告", heading))
        display = bool(re.search(r"显示屏.*(?:错误|代码)|错误代码.*显示屏", heading))
        domain = (
            "network"
            if network and not display
            else ("device_display_errors" if display and not network else "unknown")
        )
    return {
        "config": deepcopy(config),
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "document_name": name,
        "parent_id": parent.get("parent_id"),
        "heading_path": parent.get("heading_path") or ([heading] if heading else []),
        "domain": domain,
        "scope": scope,
        "source": {
            key: deepcopy(chunk[key])
            for key in (
                "parent_child",
                "heading",
                "positions",
                "block_ids",
                "source_spans",
            )
            if key in chunk
        },
    }


def _models(record: dict, default: list[str], entities: dict) -> list[str]:
    result = set(default)
    known = bool(default)
    # A named model endpoint is itself explicit fact scope, not a capability
    # inferred from the filename. Do not spread SUPPORTS to sibling models.
    endpoints = {
        entities[name]["name"].upper()
        for name in (record["source"], record["target"])
        if entities[name]["type"] == "DeviceModel"
    }
    if endpoints:
        result = result.intersection(endpoints) if known else endpoints
        known = True
    for qualifier in record["qualifiers"]:
        if qualifier["kind"] != "model_restriction":
            continue
        quote = "\n".join(qualifier["evidence_quotes"])
        found = set(MODEL_RE.findall(quote.upper().replace("_", " ")))
        exclusive = bool(
            re.search(
                r"仅(?:适用于?|对于|限于?)|只适用于?|\bonly\b", quote, re.IGNORECASE
            )
        )
        excluded = bool(
            re.search(r"不适用|除外|以外|之外|\bexcept\b", quote, re.IGNORECASE)
        )
        if not found or exclusive == excluded:
            raise ValueError(
                "Local model restriction has no unambiguous supported interpretation"
            )
        if default and not found.issubset(set(default)):
            raise ValueError(
                "Local model restriction conflicts with document/parent scope"
            )
        if excluded:
            if not known:
                raise ValueError("Excluded models require a known document model scope")
            result.difference_update(found)
        else:
            result = result.intersection(found) if known else found
            known = True
    if known and not result:
        raise ValueError("Local model restrictions have an empty intersection")
    return sorted(result)


def scoped_name(entity: dict, context: dict, models: list[str]) -> str:
    # Unknown scenes and non-code objects remain local. A missing model scope
    # must never become a globally shared "all models" identity.
    local = context["parent_id"] or context["chunk_id"]
    if entity["type"] == "FaultCode" and context["domain"] != "unknown" and models:
        local = None
    identity = [
        VERSION,
        context["doc_id"],
        context["domain"],
        local,
        entity["type"],
        entity["name"],
        sorted(models),
    ]
    return f"{entity['name']}{IDENTITY_MARKER}{fingerprint(identity)[:24]}⟧"


def skipped_evidence(context: dict, review: dict) -> dict:
    """Retain an explicit extraction gap alongside its original source ownership."""
    assert_contract(context["config"])
    return {
        "version": VERSION,
        "context": deepcopy(context),
        "nodes": {},
        "facts": [],
        "review": {**deepcopy(review), "skipped": True},
        "semantic_review_required": True,
    }


def build_records(
    content: str,
    context: dict,
    result: str,
    timestamp: int,
    *,
    on_schema_error: str = "raise",
    response_truncated: bool = False,
) -> tuple[dict, dict, dict]:
    """Validate graph candidates; optionally retain unusable responses as skipped evidence."""
    if on_schema_error not in {"raise", "skip"}:
        raise ValueError("Unknown business extraction schema error policy")
    assert_contract(context["config"])
    review = printer.review_response(
        {
            "choices": [
                {
                    "finish_reason": "length" if response_truncated else "stop",
                    "message": {"content": result},
                }
            ]
        },
        {"input_text": content},
        record_limit_policy="quarantine",
    )
    if review["sample_error"]:
        if on_schema_error == "skip":
            return {}, {}, skipped_evidence(context, review)
        raise ValueError(
            f"Business extraction output is unusable: {review['sample_error']}"
        )
    nodes, edges = defaultdict(list), defaultdict(list)
    catalog, facts, used = {}, [], set()
    entities = {e["name"]: e for e in review["accepted"]["entities"]}
    default = context["scope"]["applicable_models"]

    def node(entity, models):
        key = scoped_name(entity, context, models)
        if key not in catalog:
            catalog[key] = {
                "display_name": entity["name"],
                "type": entity["type"],
                "models": models,
                "domain": context["domain"],
                "doc_id": context["doc_id"],
            }
            nodes[key].append(
                {
                    "entity_name": key,
                    "entity_type": entity["type"].upper(),
                    "description": entity["description"],
                    "source_id": context["chunk_id"],
                    "file_path": context["document_name"],
                    "timestamp": timestamp,
                }
            )
        used.add(entity["name"])
        return key

    valid_relationships = []
    for index, relation in enumerate(review["accepted"]["relationships"]):
        try:
            models = _models(relation, default, entities)
        except ValueError as exc:
            review["rejected"].append(
                {
                    "path": f"accepted.relationships[{index}]",
                    "reason": str(exc),
                    "original_record": deepcopy(relation),
                }
            )
            continue
        left = node(entities[relation["source"]], models)
        right = node(entities[relation["target"]], models)
        detail = relation["description"]
        if relation["qualifiers"]:
            detail += "\n" + "\n".join(
                f"{q['kind']}: {q['text']}" for q in relation["qualifiers"]
            )
        edges[(left, right)].append(
            {
                "src_id": left,
                "tgt_id": right,
                "description": f"{relation['source']} --{relation['relation']}--> {relation['target']}: {detail}",
                "keywords": relation["relation"],
                "weight": 1.0,
                "source_id": context["chunk_id"],
                "file_path": context["document_name"],
                "timestamp": timestamp,
            }
        )
        facts.append(
            {
                "source_node": left,
                "target_node": right,
                "models": models,
                "record": deepcopy(relation),
            }
        )
        valid_relationships.append(relation)
    review["accepted"]["relationships"] = valid_relationships
    for entity in entities.values():
        if entity["name"] not in used:
            node(entity, sorted(default))
    return (
        nodes,
        edges,
        {
            "version": VERSION,
            "context": deepcopy(context),
            "nodes": catalog,
            "facts": facts,
            "review": review,
            "semantic_review_required": True,
        },
    )
