"""Build auditable child drafts inside immutable parent boundaries.

Callers supply a local tokenizer. This module makes no model or service calls.
JSON tables with explicit source headers may use complete-row projections;
other protected markup stays atomic. No cell or note is truncated. Coordinates
describe source fragments, not precise child rectangles.
See ``docs/ParentChildOfflinePreparation-zh.md`` for the draft contract.
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from lightrag.chunker.recursive_character import chunking_by_recursive_character
from lightrag.utils import Tokenizer

from .parents import VERSION as PARENT_VERSION
from .parents import file_hash, read_sidecar, stable_id, validate_coverage
from .table_notes import (
    notes_for_child,
    notes_for_parent,
    validate_notes,
    with_table_notes,
)
from .tables import parent_footnotes, slice_json_table, validate_table_part

VERSION = "parent-child-draft-v3"
VOCAB_SHA256 = "sha256:223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
CL100K_PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
ATOMIC_RE = re.compile(
    r"<table\b[^>]*>.*?</table>|<drawing\b[^>]*?/\s*>|<equation\b[^>]*>.*?</equation>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ChildConfig:
    target_tokens: int = 500
    overlap_tokens: int = 60

    def __post_init__(self) -> None:
        if self.target_tokens < 1 or not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("Require target_tokens > overlap_tokens >= 0")


DEFAULT_CONFIG = ChildConfig()


def load_local_tokenizer(path: Path) -> Tokenizer:
    """Load the pinned cl100k vocabulary from disk, without a cache download."""
    import tiktoken

    if file_hash(path) != VOCAB_SHA256:
        raise ValueError("Local cl100k_base vocabulary SHA-256 mismatch")
    ranks = {}
    for line in path.read_bytes().splitlines():
        encoded, rank = line.split()
        ranks[base64.b64decode(encoded, validate=True)] = int(rank)
    encoding = tiktoken.Encoding(
        name="cl100k_base",
        pat_str=CL100K_PATTERN,
        mergeable_ranks=ranks,
        special_tokens={
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    )
    return Tokenizer("cl100k_base", encoding)


def heading_conditions(parent: dict) -> dict:
    evidence, platforms = [], []
    for title in parent["heading_path"]:
        matched = [
            name
            for name, pattern in [
                ("windows", r"Windows"),
                ("macos", r"Mac\s*OS|macOS"),
                ("ios", r"iOS"),
                ("android", r"Android"),
            ]
            if re.search(
                r"(?<![A-Za-z])(?:" + pattern + r")(?![A-Za-z])", title, re.IGNORECASE
            )
        ]
        if matched:
            platforms = matched
            evidence.append(
                {"source": "heading_path", "quote": title, "platforms": matched}
            )
    return {
        "platforms": platforms,
        "evidence": evidence,
        "coverage": "heading_path_only",
    }


def atomic_spans(text: str) -> list[tuple[int, int, str]]:
    spans = [
        (m.start(), m.end(), m[0].split()[0][1:]) for m in ATOMIC_RE.finditer(text)
    ]
    for opener in re.finditer(r"<(?:table|drawing|equation)\b", text):
        if not any(start <= opener.start() < end for start, end, _ in spans):
            raise ValueError("Incomplete protected markup in parent")
    return spans


def child_spans(
    parent: dict, tokenizer: Tokenizer, config: ChildConfig
) -> list[tuple[int, int, str]]:
    text = parent["content"]
    assets = atomic_spans(text)
    if not text.strip():
        return []
    if len(tokenizer.encode(text)) <= config.target_tokens:
        return [(0, len(text), "mixed" if assets else "text")]
    result, cursor = [], 0
    for start, end, kind in [*assets, (len(text), len(text), "end")]:
        prose = text[cursor:start]
        if prose.strip():
            chunks = chunking_by_recursive_character(
                tokenizer,
                prose,
                config.target_tokens,
                chunk_overlap_token_size=config.overlap_tokens,
                separators=["\n\n", "\n", "。", "；", " ", ""],
            )
            for chunk in chunks:
                span = chunk["_source_span"]
                result.append((cursor + span["start"], cursor + span["end"], "text"))
        if end > start:
            result.append((start, end, kind))
        cursor = end
    return result


def source_refs(parent: dict, start: int, end: int) -> tuple[list[dict], list[dict]]:
    refs, positions, cursor = [], [], 0
    by_fragment = {item["fragment_id"]: item for item in parent["source_positions"]}
    for span in parent["source_spans"]:
        length = span["end"] - span["start"]
        left, right = max(start, cursor), min(end, cursor + length)
        if left < right:
            refs.append(
                {
                    "block_id": span["block_id"],
                    "fragment_id": span["fragment_id"],
                    "start": span["start"] + left - cursor,
                    "end": span["start"] + right - cursor,
                }
            )
            positions.append(deepcopy(by_fragment[span["fragment_id"]]))
        cursor += length + 2
    if not refs and parent["content"][start:end].strip():
        raise ValueError("Child content has no source attribution")
    return refs, positions


def _build_initial_children(
    parents: list[dict], tokenizer: Tokenizer, config: ChildConfig = DEFAULT_CONFIG
) -> tuple[list[dict], list[dict]]:
    """Return annotated parent copies and children; never mutate input parents."""
    annotated, children = deepcopy(parents), []
    ids = [p["parent_id"] for p in annotated]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate parent ID")
    for parent in annotated:
        parent["conditions"] = heading_conditions(parent)
        if parent["content_role"] != "body":
            continue
        for order, (start, end, kind) in enumerate(
            child_spans(parent, tokenizer, config)
        ):
            content = parent["content"][start:end]
            refs, positions = source_refs(parent, start, end)
            tokens = len(tokenizer.encode(content))
            flags = list(parent["review_flags"])
            if tokens > config.target_tokens:
                flags.append(
                    "oversized_atomic_asset" if kind != "text" else "text_above_target"
                )
            assets = []
            for match in ATOMIC_RE.finditer(content):
                identifier = re.search(r'(?<![\w-])id="([^"]+)"', match[0])
                assets.append(
                    {
                        "kind": match[0].split()[0][1:],
                        "id": identifier[1] if identifier else None,
                    }
                )
            metadata = {
                key: deepcopy(parent[key])
                for key in [
                    "doc_id",
                    "document_name",
                    "heading_path",
                    "scope",
                    "conditions",
                    "feature_domain",
                    "context_parent_ids",
                ]
            }
            retrieval_text = (
                f"Document: {parent['document_name']}\n"
                f"Section: {' / '.join(parent['heading_path'])}\n"
                f"Models: {', '.join(parent['scope']['applicable_models'])}\n"
                f"Platforms: {', '.join(parent['conditions']['platforms']) or 'unspecified'}\n\n{content}"
            )
            children.append(
                {
                    "schema_version": VERSION,
                    "child_id": stable_id(
                        "child-",
                        VERSION,
                        parent["parent_id"],
                        parent["content"],
                        metadata,
                        start,
                        end,
                        asdict(config),
                        tokenizer.model_name,
                    ),
                    "parent_id": parent["parent_id"],
                    **metadata,
                    "order_in_parent": order,
                    "kind": kind,
                    "content": content,
                    "parent_span": {"start": start, "end": end},
                    "source_spans": refs,
                    "source_positions": positions,
                    "position_precision": "source_fragment",
                    "pages": sorted(
                        {
                            int(pos["anchor"])
                            for item in positions
                            for pos in item["positions"]
                            if str(pos.get("anchor", "")).isdigit()
                        }
                    ),
                    "assets": assets,
                    "tokens": tokens,
                    "tokenizer": tokenizer.model_name,
                    "retrieval_text": retrieval_text,
                    "retrieval_tokens": len(tokenizer.encode(retrieval_text)),
                    "review_flags": sorted(set(flags)),
                    "ready_for_indexing": False,
                }
            )
    return annotated, children


def projected_source_refs(parent: dict, child: dict) -> tuple[list[dict], list[dict]]:
    span, meta = child["parent_span"], child.get("transformation")
    if meta is None:
        return source_refs(parent, span["start"], span["end"])
    intervals = sorted(
        [*meta["row_spans"], *meta["wrapper_spans"]], key=lambda row: row["start"]
    )
    refs, positions = [], {}
    for item in intervals:
        part_refs, part_positions = source_refs(
            parent, span["start"] + item["start"], span["start"] + item["end"]
        )
        refs.extend(part_refs)
        positions.update({p["fragment_id"]: p for p in part_positions})
    return refs, list(positions.values())


def build_children(
    parents: list[dict],
    tokenizer: Tokenizer,
    config: ChildConfig = DEFAULT_CONFIG,
    *,
    table_headers: dict[str, list] | None = None,
    table_sidecar: dict | None = None,
    table_sidecar_sha256: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build children and optional complete-row table projections."""
    parents = deepcopy(parents)
    if table_sidecar is not None:
        if not table_sidecar_sha256:
            raise ValueError("Table notes require a source sidecar hash")
        for parent in parents:
            parent["table_notes"] = notes_for_parent(
                parent, table_sidecar, table_sidecar_sha256
            )
    annotated, initial = _build_initial_children(parents, tokenizer, config)
    by_id, children, orders = {p["parent_id"]: p for p in annotated}, [], Counter()
    for child in initial:
        identifier = child["assets"][0]["id"] if child["assets"] else None
        header = (table_headers or {}).get(identifier)
        parts = (
            slice_json_table(child["content"], header, tokenizer, config.target_tokens)
            if child["kind"] == "table"
            and child["tokens"] > config.target_tokens
            and header
            else []
        )
        for part in parts or [{"content": child["content"], "transformation": None}]:
            item, parent = deepcopy(child), by_id[child["parent_id"]]
            item.update(part)
            item["required_notes"] = (
                parent_footnotes(parent["content"], item["content"])
                if part["transformation"]
                else []
            )
            for note in item["required_notes"]:
                note["source_spans"], note["source_positions"] = source_refs(
                    parent, note["parent_span"]["start"], note["parent_span"]["end"]
                )
            if part["transformation"]:
                item["kind"] = "table_slice"
                item["source_spans"], item["source_positions"] = projected_source_refs(
                    parent, item
                )
                item["tokens"] = len(tokenizer.encode(item["content"]))
                item["review_flags"] = [
                    f for f in item["review_flags"] if f != "oversized_atomic_asset"
                ]
                if item["tokens"] > config.target_tokens:
                    item["review_flags"].append("oversized_table_row")
                missing = set(re.findall(r"\*\s*(\d+)", item["content"])) - {
                    n["marker"] for n in item["required_notes"]
                }
                if missing:
                    item["review_flags"].append("table_footnote_not_located")
                prefix = child["retrieval_text"][: -len(child["content"])]
                item["retrieval_text"] = prefix + item["content"]
                if item["required_notes"]:
                    item["retrieval_text"] += (
                        "\n\nTable notes from parent:\n"
                        + "\n".join(n["content"] for n in item["required_notes"])
                    )
                item["retrieval_tokens"] = len(tokenizer.encode(item["retrieval_text"]))
                item["child_id"] = stable_id(
                    "child-", VERSION, child["child_id"], part, item["required_notes"]
                )
            item["table_notes"] = notes_for_child(parent, item)
            item["retrieval_text"] = with_table_notes(
                item["retrieval_text"], item["table_notes"]
            )
            item["retrieval_tokens"] = len(tokenizer.encode(item["retrieval_text"]))
            if item["table_notes"]:
                item["child_id"] = stable_id(
                    "child-", VERSION, item["child_id"], item["table_notes"]
                )
            item["order_in_parent"] = orders[item["parent_id"]]
            orders[item["parent_id"]] += 1
            children.append(item)
    validate_children(annotated, children, tokenizer, config)
    return annotated, children


def validate_children(
    parents: list[dict],
    children: list[dict],
    tokenizer: Tokenizer,
    config: ChildConfig = DEFAULT_CONFIG,
) -> dict:
    owners = {p["parent_id"]: p for p in parents}
    for parent in parents:
        validate_notes(parent)
    if len(owners) != len(parents):
        raise ValueError("Duplicate parent ID")
    seen, grouped = set(), {key: [] for key in owners}
    for child in children:
        if child["child_id"] in seen or child["parent_id"] not in owners:
            raise ValueError("Duplicate child or orphan child")
        seen.add(child["child_id"])
        parent = owners[child["parent_id"]]
        if parent["content_role"] != "body":
            raise ValueError("Non-body parent was indexed")
        for key in [
            "doc_id",
            "document_name",
            "heading_path",
            "scope",
            "conditions",
            "feature_domain",
            "context_parent_ids",
        ]:
            if child[key] != parent[key]:
                raise ValueError(f"Child inherited metadata mismatch: {key}")
        start, end = child["parent_span"]["start"], child["parent_span"]["end"]
        if not 0 <= start < end <= len(parent["content"]):
            raise ValueError("Child is not an exact parent substring")
        if child.get("transformation"):
            validate_table_part(parent["content"][start:end], child)
        elif parent["content"][start:end] != child["content"]:
            raise ValueError("Child is not an exact parent substring")
        refs, positions = projected_source_refs(parent, child)
        if child["source_spans"] != refs or child["source_positions"] != positions:
            raise ValueError("Child source attribution mismatch")
        if child.get("table_notes", []) != notes_for_child(parent, child):
            raise ValueError("Child table notes were lost or altered")
        if child["retrieval_tokens"] != len(tokenizer.encode(child["retrieval_text"])):
            raise ValueError("Retrieval token count mismatch")
        for note in child.get("table_notes", []):
            # Raw note bytes stay in table_notes even if already in source
            # body with different line-wrap whitespace.
            if re.sub(r"\s+", "", note["content"]) not in re.sub(
                r"\s+", "", child["retrieval_text"]
            ):
                raise ValueError("Table note missing from embedding text")
        tokens = len(tokenizer.encode(child["content"]))
        if tokens != child["tokens"]:
            raise ValueError("Child token count mismatch")
        if tokens > config.target_tokens and not {
            "oversized_atomic_asset",
            "text_above_target",
            "oversized_table_row",
        }.intersection(child["review_flags"]):
            raise ValueError("Oversized child is not flagged")
        grouped[child["parent_id"]].append(child)
        expected_notes = (
            parent_footnotes(parent["content"], child["content"])
            if child.get("transformation")
            else []
        )
        observed_notes = [
            {k: note[k] for k in ("kind", "marker", "parent_span", "content")}
            for note in child.get("required_notes", [])
        ]
        if expected_notes != observed_notes:
            raise ValueError("Required table footnotes were lost or altered")
        for note in child.get("required_notes", []):
            span = note["parent_span"]
            if parent["content"][span["start"] : span["end"]] != note["content"]:
                raise ValueError("Table footnote content mismatch")
            note_refs, note_positions = source_refs(parent, span["start"], span["end"])
            if (
                note["source_spans"] != note_refs
                or note["source_positions"] != note_positions
            ):
                raise ValueError("Table footnote source mismatch")
    for key, parent in owners.items():
        text, cursor = parent["content"], 0
        source_length = sum(s["end"] - s["start"] for s in parent["source_spans"])
        if source_length + 2 * (len(parent["source_spans"]) - 1) != len(text):
            raise ValueError("Parent source layout length mismatch")
        for i, child in enumerate(grouped[key]):
            start, end = child["parent_span"]["start"], child["parent_span"]["end"]
            if child["order_in_parent"] != i or (
                i and start < grouped[key][i - 1]["parent_span"]["start"]
            ):
                raise ValueError("Child order mismatch")
            if text[cursor:start].strip():
                raise ValueError("Uncovered parent content")
            cursor = max(cursor, end)
        if parent["content_role"] == "body" and text[cursor:].strip():
            raise ValueError("Uncovered parent content")
        for start, end, _ in atomic_spans(text):
            slices = [
                c
                for c in grouped[key]
                if c.get("transformation")
                and c["parent_span"] == {"start": start, "end": end}
            ]
            if slices:
                row_cursor = slices[0]["transformation"]["header_count"]
                for item in slices:
                    meta = item["transformation"]
                    if (
                        meta["row_start"] != row_cursor
                        or meta["header_count"]
                        != slices[0]["transformation"]["header_count"]
                    ):
                        raise ValueError("Table rows dropped, duplicated or reordered")
                    row_cursor = meta["row_end"]
                if row_cursor != slices[-1]["transformation"]["total_rows"]:
                    raise ValueError("Uncovered table rows")
            if parent["content_role"] == "body" and not any(
                c["parent_span"]["start"] <= start and c["parent_span"]["end"] >= end
                for c in grouped[key]
            ):
                raise ValueError("Protected markup was split across children")
    return {
        "body_parents": sum(p["content_role"] == "body" for p in parents),
        "children": len(children),
        "nonwhitespace_content_covered": True,
        "exact_parent_substrings": not any(c.get("transformation") for c in children),
        "verified_source_projections": True,
        "table_rows_verified": True,
        "source_attribution_verified": True,
        "table_note_count": sum(len(p.get("table_notes", [])) for p in parents),
        "protected_markup_preserved": True,
    }


def read_parent_bundle(directory: Path) -> tuple[list[dict], dict]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["schema_version"] != PARENT_VERSION:
        raise ValueError("Unsupported parent draft version")
    parents = [
        json.loads(line)
        for line in (directory / "parents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(parents) != summary["parents"] or any(
        p["doc_id"] != summary["doc_id"]
        or p["document_name"] != summary["document_name"]
        for p in parents
    ):
        raise ValueError("Parent bundle identity/count mismatch")
    source = Path(summary["inputs"]["sidecar"])
    if file_hash(source) != summary["inputs"]["sidecar_sha256"]:
        raise ValueError("Source sidecar hash mismatch")
    _, blocks = read_sidecar(source)
    validate_coverage(blocks, parents)
    return parents, summary


def prepare_children(
    parent_dir: Path,
    output: Path,
    tokenizer: Tokenizer,
    config: ChildConfig = DEFAULT_CONFIG,
    *,
    tokenizer_sha256: str | None = None,
    table_header_overrides: Path | None = None,
) -> dict:
    """Write a new offline bundle; summary.json is written last."""
    if output.exists():
        raise ValueError("Output already exists; choose a new experiment directory")
    parents, source_summary = read_parent_bundle(parent_dir)
    sidecar = Path(source_summary["inputs"]["sidecar"])
    tables_path = sidecar.with_name(
        sidecar.name.replace(".blocks.jsonl", ".tables.json")
    )
    headers, tables = {}, None
    if tables_path.exists():
        tables = json.loads(tables_path.read_text(encoding="utf-8"))["tables"]
        for key, table in tables.items():
            if table.get("format") == "json" and table.get("table_header"):
                headers[key] = json.loads(table["table_header"])
    if table_header_overrides:
        overrides = json.loads(table_header_overrides.read_text(encoding="utf-8"))
        if overrides.get("schema_version") != 1:
            raise ValueError("Unsupported table header override schema")
        for key, item in overrides["tables"].items():
            if item["evidence"]["document_name"] == source_summary["document_name"]:
                if not item["header_rows"] or not item["evidence"].get("pdf_page"):
                    raise ValueError("Table header override has no evidence")
                headers[key] = item["header_rows"]
    parents, children = build_children(
        parents,
        tokenizer,
        config,
        table_headers=headers,
        table_sidecar=tables,
        table_sidecar_sha256=file_hash(tables_path) if tables is not None else None,
    )
    checks = validate_children(parents, children, tokenizer, config)
    summary = {
        "schema_version": VERSION,
        "doc_id": source_summary["doc_id"],
        "document_name": source_summary["document_name"],
        **checks,
        "parents": len(parents),
        "config": asdict(config),
        "tokenizer": tokenizer.model_name,
        "tokenizer_sha256": tokenizer_sha256,
        "ready_for_indexing": False,
        "review_counts": dict(
            Counter(flag for c in children for flag in c["review_flags"])
        ),
        "max_content_tokens": max((c["tokens"] for c in children), default=0),
        "max_retrieval_tokens": max(
            (c["retrieval_tokens"] for c in children), default=0
        ),
        "inputs": {
            "parent_dir": str(parent_dir.resolve()),
            "parents_sha256": file_hash(parent_dir / "parents.jsonl"),
            "parent_summary_sha256": file_hash(parent_dir / "summary.json"),
            "asset_base_dir": str(Path(source_summary["inputs"]["sidecar"]).parent),
            "table_sidecar_sha256": file_hash(tables_path)
            if tables_path.exists()
            else None,
            "table_sidecar": str(tables_path.resolve()) if tables is not None else None,
            "table_header_overrides_sha256": file_hash(table_header_overrides)
            if table_header_overrides
            else None,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [("parents.jsonl", parents), ("children.jsonl", children)]:
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    summary["outputs"] = {
        name: file_hash(output / name) for name in ("parents.jsonl", "children.jsonl")
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
