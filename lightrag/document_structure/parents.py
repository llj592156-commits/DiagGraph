"""Prepare auditable parent-block candidates from a PDF and existing sidecars.

This shared organizer is not an ingestion path. It never rewrites source
PDFs, sidecars, or a knowledge base and makes no model or service requests.
Use a new output directory for each experiment. Uncertain alignments and large
units require review; their presence is not evidence of indexing readiness.
See ``docs/ParentBlockPreparation-zh.md`` for the preparation contract.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "parent-block-draft-v3"
MODEL_RE = re.compile(r"\b[A-Z]{1,6}\d{3,6}\b")
FAULT_RE = re.compile(r"[A-Z]{1,3}-?\d{1,4}(?:\s*[/、,]\s*[A-Z]{1,3}-?\d{1,4})*")
QUALIFIER_RE = re.compile(r"仅(?:对于|适用于?|限)|只适用于?|适用型号")
SUPPORT_HEADINGS = {"解决方法", "处理方法", "相关信息", "注意", "重要", "注", "警告"}


@dataclass(frozen=True)
class ParentPreparation:
    """In-memory document structure for CLIs and future ingestion adapters."""

    metadata: dict
    sections: list[dict]
    parents: list[dict]
    issues: list[dict]
    summary: dict


@dataclass(frozen=True)
class Bookmark:
    section_id: str
    parent_id: str | None
    path: tuple[str, ...]
    page: int
    top: float | None


def stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def normalize_title(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()


def normalize_quote(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return "sha256:" + digest.hexdigest()


def read_sidecar(path: Path) -> tuple[dict, list[dict]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("type") != "meta":
        raise ValueError("The first sidecar row must be metadata")
    meta, blocks = rows[0], rows[1:]
    if not blocks or meta.get("blocks") != len(blocks):
        raise ValueError("Sidecar block count does not match metadata")
    seen = set()
    for block in blocks:
        key = block.get("blockid")
        if block.get("type") != "content" or not isinstance(key, str) or not key:
            raise ValueError("Each content row must have a nonempty blockid")
        if key in seen:
            raise ValueError(f"Duplicate source blockid: {key}")
        if not isinstance(block.get("content"), str):
            raise TypeError(f"Invalid content in block {key}")
        seen.add(key)
    if meta.get("document_hash"):
        merged = "\n\n".join(block["content"] for block in blocks if block["content"])
        digest = "sha256:" + hashlib.sha256(merged.encode("utf-8")).hexdigest()
        if digest != meta["document_hash"]:
            raise ValueError("Sidecar content does not match metadata document_hash")
    return meta, blocks


def read_pdf(path: Path, doc_id: str) -> tuple[list[Bookmark], dict[int, float], Any]:
    """Read original outline destinations; no text or layout is inferred."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    heights = {
        i + 1: float(page.mediabox.height) for i, page in enumerate(reader.pages)
    }
    entries: list[Bookmark] = []

    def walk(nodes: list, parent: Bookmark | None = None) -> None:
        previous = None
        for node in nodes:
            if isinstance(node, list):
                if previous is None:
                    raise ValueError("Outline children have no preceding destination")
                walk(node, previous)
                continue
            page = reader.get_destination_page_number(node)
            if page is None or page < 0:
                raise ValueError(f"Unresolved outline destination: {node.title}")
            title = str(node.title)
            path_titles = (parent.path if parent else ()) + (title,)
            top = node.get("/Top")
            top = float(top) if isinstance(top, (float, int)) else None
            entry = Bookmark(
                stable_id("section-", doc_id, len(entries), path_titles, page, top),
                parent.section_id if parent else None,
                path_titles,
                page + 1,
                top,
            )
            entries.append(entry)
            previous = entry

    walk(reader.outline)
    if not entries:
        raise ValueError(
            "PDF has no outline; this prototype requires original bookmarks"
        )
    return entries, heights, reader


def block_start(
    block: dict, heights: dict[int, float], origin: str
) -> tuple[int, float] | None:
    positions = block.get("positions") or []
    if not positions:
        return None
    # The first position belongs to the heading, rather than the earliest box
    # of a cross-page block. Source order is preserved; boxes are never sorted.
    pos = positions[0]
    try:
        page = int(pos["anchor"])
        box = pos["range"]
        if pos.get("type") != "bbox" or len(box) != 4:
            return None
        if origin == "LEFTBOTTOM":
            return page, heights[page] - max(float(box[1]), float(box[3]))
        if origin == "LEFTTOP":
            return page, min(float(box[1]), float(box[3]))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def align_sections(
    bookmarks: list[Bookmark],
    blocks: list[dict],
    heights: dict[int, float],
    origin: str,
) -> tuple[list[dict], list[dict]]:
    """Prefer title plus page; position-only fallback is explicitly uncertain."""
    starts = [block_start(block, heights, origin) for block in blocks]
    sections, issues = [], []
    previous_index = 0
    for bookmark in bookmarks:
        target_y = (
            heights[bookmark.page] - bookmark.top if bookmark.top is not None else 0.0
        )
        candidates = [
            i
            for i, block in enumerate(blocks)
            if i >= previous_index
            and starts[i] is not None
            and starts[i][0] == bookmark.page
            and normalize_title(block.get("heading", ""))
            == normalize_title(bookmark.path[-1])
        ]
        method = "title_and_page"
        if candidates:
            index = min(candidates, key=lambda i: abs(starts[i][1] - target_y))
            if blocks[index].get("bookmark_evidence"):
                method = "picture_text_and_page"
        else:
            method = "position_only"
            index = next(
                (
                    i
                    for i, start in enumerate(starts)
                    if i >= previous_index
                    and start is not None
                    and bookmark.page <= start[0] <= bookmark.page + 1
                    and start >= (bookmark.page, target_y - 12)
                ),
                None,
            )
            if index is None:
                method = "unmatched"
            issues.append(
                {
                    "kind": "outline_alignment_requires_review",
                    "section_id": bookmark.section_id,
                    "heading_path": list(bookmark.path),
                    "page": bookmark.page,
                    "method": method,
                }
            )
        if index is not None:
            previous_index = index
        sections.append(
            {
                "section_id": bookmark.section_id,
                "parent_section_id": bookmark.parent_id,
                "heading_path": list(bookmark.path),
                "level": len(bookmark.path),
                "pdf_page": bookmark.page,
                "pdf_top": bookmark.top,
                "alignment": method,
                "start_block_index": index,
                "start_block_id": blocks[index]["blockid"]
                if index is not None
                else None,
                "evidence": blocks[index].get("bookmark_evidence")
                if index is not None
                else None,
            }
        )
    return sections, issues


def split_picture_bookmarks(
    blocks: list[dict],
    bookmarks: list[Bookmark],
    raw: dict,
    drawings: dict,
    heights: dict[int, float],
    origin: str,
) -> tuple[list[dict], list[dict]]:
    """Split only at a picture with matching embedded title and provenance.

    Offsets refer to the original sidecar content, never Docling charspan.
    Ambiguous or cross-position evidence leaves the original block intact.
    """
    pictures = {item["self_ref"]: item for item in raw.get("pictures", [])}
    texts = {item["self_ref"]: item for item in raw.get("texts", [])}
    result, events = [], []
    for block in blocks:
        cuts = []
        start = block_start(block, heights, origin)
        for match in re.finditer(r'<drawing id="([^"]+)"[^>]*\s/>', block["content"]):
            drawing = drawings.get(match[1], {})
            if drawing.get("blockid") != block["blockid"]:
                continue
            picture = pictures.get(drawing.get("self_ref"), {})
            prov = picture.get("prov", [])
            if len(prov) != 1:
                continue
            page = prov[0].get("page_no")
            box = prov[0].get("bbox", {})
            if box.get("coord_origin") != "BOTTOMLEFT":
                continue
            position = {
                "type": "bbox",
                "anchor": str(page),
                "range": [box[k] for k in ("l", "t", "r", "b")],
            }
            target = block_start({"positions": [position]}, heights, origin)
            if start is None or target is None or start[0] >= target[0]:
                continue
            children = [
                texts.get(child.get("$ref"), {})
                for child in picture.get("children", [])
            ]
            titles = [
                child.get("text", "")
                for child in children
                if child.get("content_layer", "body") == "body"
                and any(p.get("page_no") == page for p in child.get("prov", []))
            ]
            candidates = [
                b
                for b in bookmarks
                if b.page == page
                and b.top is not None
                and abs(heights[page] - b.top - target[1]) <= 12
                and any(
                    normalize_title(t) == normalize_title(b.path[-1]) for t in titles
                )
            ]
            if len(candidates) != 1 or block["content"].count(match[0]) != 1:
                continue
            # A page boundary can partition positions without pretending their
            # per-item charspans are offsets in the concatenated block.
            positions = block.get("positions", [])
            pages = [int(p["anchor"]) for p in positions]
            if pages != sorted(pages) or not any(p == page for p in pages):
                continue
            first = next(p for p in positions if int(p["anchor"]) == page)
            if first.get("range") != position["range"]:
                continue
            evidence = {
                "section_id": candidates[0].section_id,
                "heading_path": list(candidates[0].path),
                "page": page,
                "raw_picture_ref": picture["self_ref"],
                "embedded_title": candidates[0].path[-1],
                "drawing_id": match[1],
            }
            cuts.append((match.start(), page, evidence))
        if not cuts:
            result.append(block)
            continue
        cuts.sort()
        boundaries = [(0, None, None), *cuts, (len(block["content"]), None, None)]
        for i, (offset, page, evidence) in enumerate(boundaries[:-1]):
            end, next_page, _ = boundaries[i + 1]
            fragment = {
                **block,
                "blockid": stable_id("fragment-", block["blockid"], offset, end),
                "content": block["content"][offset:end],
                "source_span": {
                    "block_id": block["blockid"],
                    "start": offset,
                    "end": end,
                },
                "positions": [
                    p
                    for p in block.get("positions", [])
                    if (page is None or int(p["anchor"]) >= page)
                    and (next_page is None or int(p["anchor"]) < next_page)
                ],
            }
            if evidence:
                fragment["heading"] = evidence["embedded_title"]
                fragment["bookmark_evidence"] = evidence
                events.append(
                    {
                        "kind": "picture_bookmark_boundary_recovered",
                        "source_block_id": block["blockid"],
                        "offset": offset,
                        **evidence,
                    }
                )
            result.append(fragment)
    return result, events


def load_structure_rules(path: Path | None, document_name: str) -> list[dict]:
    if path is None:
        return []
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported structure rule schema")
    if document_name not in config.get("document_names", []):
        return []
    rules = config.get("rules", [])
    seen = set()
    for rule in rules:
        if not rule.get("id") or rule["id"] in seen or not rule.get("section_path"):
            raise ValueError(
                "Structure rules require unique IDs and exact section paths"
            )
        seen.add(rule["id"])
        styles = rule.get("heading_styles", [])
        if not styles or any(
            s.get("bbox_height_pt", 0) <= 0 or s.get("depth", 0) < 1 for s in styles
        ):
            raise ValueError("Structure styles require positive heights and depths")
    return rules


def refine_units(units: list, rules: list[dict], max_chars: int) -> list:
    """Refine overlong reviewed manual sections at measured heading styles.

    Warnings, numbered steps, and lower-level poster instructions stay attached.
    This is opt-in for named documents and exact paths, not a generic PDF rule.
    """
    result = []
    for section, members in units:
        rule = next(
            (
                r
                for r in rules
                if section and r["section_path"] == section["heading_path"]
            ),
            None,
        )
        if rule is None or len("\n\n".join(b["content"] for b in members)) <= max_chars:
            result.append((section, members))
            continue
        stack, current, active = [], [], section
        for block in members:
            title = block.get("heading", "").strip()
            positions = block.get("positions") or []
            box = positions[0].get("range", []) if positions else []
            height = abs(box[1] - box[3]) if len(box) == 4 else 0
            style = next(
                (
                    s
                    for s in rule["heading_styles"]
                    if abs(height - s["bbox_height_pt"]) <= 0.15
                ),
                None,
            )
            if (
                style
                and not support_heading(title)
                and not re.match(r"^\d+[.、]", title)
            ):
                depth = style["depth"]
                stack = [(d, t) for d, t in stack if d < depth]
                stack.append((depth, title))
                path = list(section["heading_path"])
                for _, heading in stack:
                    if normalize_title(path[-1]) != normalize_title(heading):
                        path.append(heading)
                evidence = {
                    "rule_id": rule["id"],
                    "block_id": block["blockid"],
                    "bbox_height_pt": height,
                    "depth": depth,
                }
                # Keep bare chapter introductions with the first operation.
                bare_prefix = all(
                    b["content"].strip() == f"## {b.get('heading', '')}"
                    for b in current
                )
                if current and not bare_prefix:
                    result.append((active, current))
                    current = []
                active = {
                    **section,
                    "heading_path": path,
                    "local_path_resolved": True,
                    "structural_evidence": evidence,
                }
            current.append(block)
        if current:
            result.append((active, current))
    return result


def load_rules(path: Path | None, document_name: str, reader: Any) -> list[dict]:
    if path is None:
        return []
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported scope rule schema")
    rules = config.get("documents", {}).get(document_name, {}).get("rules", [])
    seen = set()
    for rule in rules:
        if not rule.get("id") or rule["id"] in seen:
            raise ValueError("Scope rule IDs must be present and unique")
        seen.add(rule["id"])
        if not rule.get("applicable_models") or not rule.get("path_patterns"):
            raise ValueError(f"Scope rule {rule['id']} needs models and selectors")
        for pattern in rule["path_patterns"]:
            re.compile(pattern)
        evidence = rule.get("evidence", {})
        page, quote = evidence.get("page"), evidence.get("quote")
        if not isinstance(page, int) or not 1 <= page <= len(reader.pages) or not quote:
            raise ValueError(f"Scope rule {rule['id']} has no valid evidence")
        text = reader.pages[page - 1].extract_text() or ""
        quotes = [quote]
        if evidence.get("feature_quote"):
            quotes.append(evidence["feature_quote"])
        if any(normalize_quote(value) not in normalize_quote(text) for value in quotes):
            raise ValueError(
                f"Scope rule {rule['id']} quote not found on PDF page {page}"
            )
    return rules


def qualified_models(title: str) -> list[str]:
    if not QUALIFIER_RE.search(title):
        return []
    if re.search(r"以外|除外|不适用|不包括|非|除.+外", title):
        return []
    return sorted(set(MODEL_RE.findall(title.upper())))


def support_heading(title: str) -> bool:
    title_without_marker = re.sub(r"^[A-Za-z!！⚠\s]+", "", title)
    return title_without_marker.strip(" ：:。\t") in SUPPORT_HEADINGS or bool(
        re.match(r"^\d+[.、．]\s*", title)
    )


def is_contents_block(block: dict) -> bool:
    return len(re.findall(r"\.{4,}|…{3,}", block["content"])) >= 2


def opening_model_declaration(block: dict) -> str | None:
    """Return only the first body line when it declares a whole-unit limit."""
    for line in block["content"].splitlines()[1:]:
        if not line.strip():
            continue
        if re.fullmatch(
            r"(?:本(?:章节|节|功能|菜单|操作|说明|步骤)|传真功能)?\s*"
            r"(?:仅对于|仅适用于?|只适用于?|不适用于?)\s*"
            r"[A-Za-z0-9\s/、,，和及]+[。.]?",
            line.strip(),
        ) and MODEL_RE.search(line.upper()):
            return line.strip()
        break
    return None


def has_model_limit(block: dict) -> bool:
    title = block.get("heading", "")
    return bool(
        MODEL_RE.search(title.upper())
        and (QUALIFIER_RE.search(title) or re.search(r"不适用|除外|以外的型号", title))
        or opening_model_declaration(block)
    )


def make_units(
    blocks: list[dict], sections: list[dict]
) -> list[tuple[dict | None, list[dict]]]:
    """Group within bookmark intervals, with code and local-scope boundaries."""
    anchors = [
        section for section in sections if section["start_block_index"] is not None
    ]
    anchors.sort(key=lambda section: section["start_block_index"])
    indices = [section["start_block_index"] for section in anchors]
    units: list[tuple[dict | None, list[dict]]] = []
    current: list[dict] = []
    active = None
    local_scope = False
    local_code = False
    for i, block in enumerate(blocks):
        slot = bisect_right(indices, i) - 1
        section = anchors[slot] if slot >= 0 else None
        title = block.get("heading", "").strip()
        special = bool(FAULT_RE.fullmatch(title) or has_model_limit(block))
        local_exit = (local_scope or local_code) and not support_heading(title)
        contents_boundary = is_contents_block(block) or (
            current and is_contents_block(current[-1])
        )
        if current and (
            section is not active or special or local_exit or contents_boundary
        ):
            units.append((active, current))
            current = []
            local_scope = False
            local_code = False
        if not current:
            active = section
            local_scope = has_model_limit(block)
            local_code = bool(FAULT_RE.fullmatch(title))
        current.append(block)
    if current:
        units.append((active, current))
    return units


def resolve_scope(
    path: list[str], blocks: list[dict], rules: list[dict], document_name: str = ""
) -> tuple[dict, list[str]]:
    """Use filename models by default; only explicit unit limits narrow them.

    A feature rule from another page is reference material, not permission to
    narrow this unit. Inline restrictions on individual items remain review
    notes, unless an opening declaration explicitly governs the whole unit.
    """
    defaults = set(MODEL_RE.findall(document_name.upper().replace("_", " ")))
    claims, evidence, excluded, rule_ids, features, flags = [], [], set(), [], [], []
    if defaults:
        evidence.append({"source": "filename", "quote": document_name})
    titles = list(dict.fromkeys([*path, blocks[0].get("heading", "")]))
    declaration = opening_model_declaration(blocks[0])
    declarations = [declaration] if declaration else []
    for text in dict.fromkeys([*titles, *declarations]):
        negative = re.fullmatch(
            r"(?:仅对于|仅适用于?|除)?\s*([A-Za-z0-9\s/、,，和及]+)\s*(?:以外的型号|除外|之外)[。.]?",
            text.strip(),
        ) or re.fullmatch(
            r"(?:本(?:章节|节|功能|菜单|操作|说明|步骤))?不适用于?\s*([A-Za-z0-9\s/、,，和及]+)[。.]?",
            text.strip(),
        )
        negative_models = (
            set(MODEL_RE.findall(negative[1].upper())) if negative else set()
        )
        models = qualified_models(text) if not negative_models else []
        if models or negative_models:
            if models:
                claims.append(set(models))
            excluded.update(negative_models)
            evidence.append(
                {
                    "source": "explicit_declaration"
                    if text in declarations
                    else "explicit_heading",
                    "quote": text,
                    "block_id": blocks[0]["blockid"]
                    if text in declarations or text == blocks[0].get("heading")
                    else None,
                }
            )
    for rule in rules:
        if any(
            re.search(pattern, title)
            for pattern in rule["path_patterns"]
            for title in path
        ):
            rule_ids.append(rule["id"])
            features.append(rule.get("feature", ""))
    applicable = set(defaults)
    if claims:
        applicable = set.intersection(*([defaults] if defaults else []), *claims)
    applicable -= excluded
    explicit = bool(claims or excluded)
    status = "explicit" if explicit else "filename" if defaults else "unknown"
    if explicit:
        if not applicable and (defaults or claims):
            status = "conflict"
            flags.append("conflicting_model_scope")
        elif not defaults and not claims:
            status = "unknown"
            flags.append("excluded_models_without_filename_scope")
    return {
        "status": status,
        "policy": "filename_default_explicit_parent_limits",
        "applicable_models": sorted(applicable),
        "excluded_models": sorted(excluded),
        "evidence": evidence,
        "reference_rule_ids": rule_ids,
        "features": sorted(set(features) - {""}),
    }, flags


def build_parents(
    meta: dict,
    blocks: list[dict],
    sections: list[dict],
    rules: list[dict],
    max_chars: int,
    structure_rules: list[dict] | None = None,
) -> list[dict]:
    parents = []
    for section, members in refine_units(
        make_units(blocks, sections), structure_rules or [], max_chars
    ):
        path = list(section["heading_path"]) if section else []
        first_heading = members[0].get("heading", "").strip()
        if (
            not (section and section.get("local_path_resolved"))
            and first_heading
            and (
                not path or normalize_title(first_heading) != normalize_title(path[-1])
            )
        ):
            path.append(first_heading)
        content = "\n\n".join(block["content"] for block in members)
        source_ids = [block["blockid"] for block in members]
        scope, flags = resolve_scope(path, members, rules, meta["document_name"])
        role = "body" if section else "front_matter"
        if path and normalize_title(path[0]) in {"目录", "contents", "tableofcontents"}:
            role = "table_of_contents"
        elif all(
            is_contents_block(block) or block.get("bookmark_evidence")
            for block in members
        ):
            role = "section_contents"
        if section and section["alignment"] not in {
            "title_and_page",
            "picture_text_and_page",
        }:
            flags.append("position_only_section_alignment")
        if len(content) > max_chars:
            # Preserve a complete procedure or table rather than silently
            # cutting it. A later structural split must resolve this flag.
            flags.append("oversized_parent_requires_structural_split")
        if role == "body" and scope["status"] == "unknown":
            flags.append("model_scope_unknown")
        handled = {
            item["quote"]
            for item in scope["evidence"]
            if item["source"].startswith("explicit_")
        }
        if any(
            QUALIFIER_RE.search(line) and line.lstrip("# ").strip() not in handled
            for line in content.splitlines()
        ):
            flags.append("inline_condition_requires_review")
        pages = sorted(
            {
                int(pos["anchor"])
                for block in members
                for pos in block.get("positions", [])
                if str(pos.get("anchor", "")).isdigit()
            }
        )
        if not pages:
            flags.append("source_position_missing")
        domain = "unknown"
        if any("网络" in title for title in path):
            domain = "network"
        elif any("显示屏" in title and "错误代码" in title for title in path):
            domain = "device_display_errors"
        codes = [title for title in path if FAULT_RE.fullmatch(title)]
        parents.append(
            {
                "schema_version": VERSION,
                "parent_id": stable_id("parent-", VERSION, meta["doc_id"], source_ids),
                "doc_id": meta["doc_id"],
                "document_name": meta["document_name"],
                "document_models": sorted(
                    set(
                        MODEL_RE.findall(
                            meta["document_name"].upper().replace("_", " ")
                        )
                    )
                ),
                "document_models_source": "filename_default_scope",
                "section_id": section["section_id"] if section else None,
                "heading_path": path,
                "content_role": role,
                "feature_domain": domain,
                "fault_codes": codes,
                "scope": scope,
                "content": content,
                "char_count": len(content),
                "pages": pages,
                "source_block_ids": list(
                    dict.fromkeys(
                        block.get("source_span", {}).get("block_id", block["blockid"])
                        for block in members
                    )
                ),
                "source_fragment_ids": source_ids,
                "source_spans": [
                    {
                        "fragment_id": block["blockid"],
                        **block.get(
                            "source_span",
                            {
                                "block_id": block["blockid"],
                                "start": 0,
                                "end": len(block["content"]),
                            },
                        ),
                    }
                    for block in members
                ],
                "structural_evidence": section.get("structural_evidence")
                if section
                else None,
                "source_positions": [
                    {
                        "block_id": block.get("source_span", {}).get(
                            "block_id", block["blockid"]
                        ),
                        "fragment_id": block["blockid"],
                        "positions": block.get("positions", []),
                    }
                    for block in members
                ],
                "review_flags": sorted(set(flags)),
                "ready_for_indexing": False,
            }
        )
    for i, parent in enumerate(parents):
        path = parent["heading_path"]
        parent["context_parent_ids"] = [
            p["parent_id"]
            for p in parents[:i]
            if p["section_id"] == parent["section_id"]
            and p["content_role"] == "body"
            and len(p["heading_path"]) < len(path)
            and path[: len(p["heading_path"])] == p["heading_path"]
        ]
    return parents


def validate_coverage(blocks: list[dict], parents: list[dict]) -> dict:
    expected = [block["blockid"] for block in blocks]
    sources = {b["blockid"]: b["content"] for b in blocks}
    observed, cursor, fragments = [], 0, 0
    for parent in parents:
        spans = parent["source_spans"]
        if parent["source_fragment_ids"] != [s["fragment_id"] for s in spans] or parent[
            "source_block_ids"
        ] != list(dict.fromkeys(s["block_id"] for s in spans)):
            raise ValueError("Parent source fragment IDs disagree with spans")
        pieces = []
        for span in spans:
            key, start, end = span["block_id"], span["start"], span["end"]
            if cursor == 0:
                if len(observed) >= len(expected) or expected[len(observed)] != key:
                    raise ValueError(
                        "Parent preparation dropped, duplicated, or reordered source blocks"
                    )
                observed.append(key)
            elif observed[-1] != key:
                raise ValueError("Source fragment interleaving")
            empty = start == end == 0 and sources[key] == ""
            if start != cursor or not (empty or start < end <= len(sources[key])):
                raise ValueError("Source fragment gap or overlap")
            pieces.append(sources[key][start:end])
            cursor = 0 if end == len(sources[key]) else end
            fragments += 1
        if parent["content"] != "\n\n".join(pieces):
            raise ValueError("Parent preparation changed source content")
    if observed != expected or cursor:
        raise ValueError("Parent preparation left source content unassigned")
    return {
        "source_blocks": len(expected),
        "assigned_blocks": len(observed),
        "assigned_fragments": fragments,
        "source_order_preserved": True,
        "source_content_preserved": True,
    }


def report_crossings(
    bookmarks: list[Bookmark],
    blocks: list[dict],
    parents: list[dict],
    heights: dict[int, float],
    origin: str,
) -> list[dict]:
    """Flag indivisible source blocks that contain multiple bookmark scopes."""
    owners = {
        key: parent for parent in parents for key in parent["source_fragment_ids"]
    }
    issues = []
    for block in blocks:
        start = block_start(block, heights, origin)
        if start is None:
            continue
        positions = [
            block_start({"positions": [pos]}, heights, origin)
            for pos in block.get("positions", [])
        ]
        for bookmark in bookmarks:
            if bookmark.top is None:
                continue
            target = (bookmark.page, heights[bookmark.page] - bookmark.top)
            after = (target[0], target[1] + 12)
            if start < (target[0], target[1] - 12) and any(
                pos is not None and pos > after for pos in positions
            ):
                owner = owners[block["blockid"]]
                flag = "source_block_crosses_bookmark"
                if flag not in owner["review_flags"]:
                    owner["review_flags"].append(flag)
                    owner["review_flags"].sort()
                issues.append(
                    {
                        "kind": flag,
                        "parent_id": owner["parent_id"],
                        "block_id": block["blockid"],
                        "section_id": bookmark.section_id,
                        "heading_path": list(bookmark.path),
                        "page": bookmark.page,
                    }
                )
    return issues


def preview_markdown(summary: dict, parents: list[dict]) -> str:
    lines = [
        "# 父块候选预览",
        "",
        "这是离线草稿，尚未入库；默认适用于文件名中的型号，父块明确限定时缩小范围。",
        "",
        f"文档：{summary['document_name']}",
        f"原始结构块：{summary['source_blocks']}；父块候选：{len(parents)}。",
        "",
    ]
    for parent in parents:
        lines.extend(
            [
                f"## {' / '.join(parent['heading_path']) or '前置内容'}",
                "",
                f"- ID：`{parent['parent_id']}`",
                f"- PDF 页：{', '.join(map(str, parent['pages']))}",
                f"- 用途：{parent['content_role']}；领域：{parent['feature_domain']}",
                f"- 适用范围：{parent['scope']['status']} / {', '.join(parent['scope']['applicable_models']) or '未知'}",
                f"- 待复核：{', '.join(parent['review_flags']) or '无自动标记；仍需抽查'}",
                "",
                "<details><summary>原文（保留图片、表格引用）</summary>",
                "",
                parent["content"],
                "",
                "</details>",
                "",
            ]
        )
    return "\n".join(lines)


def organize(
    pdf: Path,
    parsed_dir: Path,
    scope_rules: Path | None = None,
    max_parent_chars: int = 10000,
    structure_rules: Path | None = None,
    *,
    document_name: str | None = None,
    doc_id: str | None = None,
) -> ParentPreparation:
    """Organize an existing parsed document without writing or indexing it."""
    if max_parent_chars < 1:
        raise ValueError("max_parent_chars must be positive")
    files = list(parsed_dir.glob("*.blocks.jsonl"))
    if len(files) != 1:
        raise ValueError("parsed-dir must contain exactly one blocks.jsonl")
    meta, blocks = read_sidecar(files[0])
    name = document_name or pdf.name
    if meta.get("document_name") != name:
        raise ValueError("PDF filename does not match sidecar document_name")
    if doc_id:
        meta = {**meta, "doc_id": doc_id}
    pdf_hash = file_hash(pdf)
    manifest = parsed_dir.parent / f"{name}.docling_raw" / "_manifest.json"
    identity_verified = False
    if manifest.is_file():
        raw_meta = json.loads(manifest.read_text(encoding="utf-8"))
        if raw_meta.get("source_content_hash") != pdf_hash:
            raise ValueError(
                "PDF bytes do not match the Docling raw manifest source hash"
            )
        identity_verified = True
    bookmarks, heights, reader = read_pdf(pdf, meta["doc_id"])
    rules = load_rules(scope_rules, name, reader)
    origin = meta.get("bbox_attributes", {}).get("origin", "LEFTBOTTOM")
    working_blocks, recovery = blocks, []
    raw_path = manifest.parent / f"{Path(name).stem}.json"
    drawings_path = files[0].with_name(
        files[0].name.replace(".blocks.jsonl", ".drawings.json")
    )
    if identity_verified and raw_path.is_file() and drawings_path.is_file():
        critical = raw_meta.get("critical_file", {})
        if critical.get("path") != raw_path.name or critical.get("sha256") != file_hash(
            raw_path
        ):
            raise ValueError("Docling raw JSON does not match manifest critical_file")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        drawings = json.loads(drawings_path.read_text(encoding="utf-8")).get(
            "drawings", {}
        )
        working_blocks, recovery = split_picture_bookmarks(
            blocks, bookmarks, raw, drawings, heights, origin
        )
    sections, issues = align_sections(bookmarks, working_blocks, heights, origin)
    issues.extend(recovery)
    if not identity_verified:
        issues.append(
            {
                "kind": "source_identity_unverified",
                "message": "No adjacent Docling raw manifest",
            }
        )
    styles = load_structure_rules(structure_rules, name)
    parents = build_parents(
        meta, working_blocks, sections, rules, max_parent_chars, styles
    )
    coverage = validate_coverage(blocks, parents)
    issues.extend(report_crossings(bookmarks, working_blocks, parents, heights, origin))
    for parent in parents:
        for flag in parent["review_flags"]:
            issues.append(
                {
                    "kind": flag,
                    "parent_id": parent["parent_id"],
                    "heading_path": parent["heading_path"],
                    "pages": parent["pages"],
                }
            )
    summary = {
        "schema_version": VERSION,
        "document_name": name,
        "doc_id": meta["doc_id"],
        **coverage,
        "pdf_pages": len(heights),
        "bookmarks": len(bookmarks),
        "alignment_counts": dict(Counter(section["alignment"] for section in sections)),
        "parents": len(parents),
        "content_roles": dict(Counter(p["content_role"] for p in parents)),
        "scope_counts": dict(Counter(p["scope"]["status"] for p in parents)),
        "review_counts": dict(
            Counter(flag for p in parents for flag in p["review_flags"])
        ),
        "source_identity_verified": identity_verified,
        "inputs": {
            "pdf": str(pdf.resolve()),
            "pdf_sha256": pdf_hash,
            "sidecar": str(files[0].resolve()),
            "sidecar_sha256": file_hash(files[0]),
            "scope_rules_sha256": file_hash(scope_rules) if scope_rules else None,
            "structure_rules_sha256": file_hash(structure_rules)
            if structure_rules
            else None,
            "raw_json_sha256": file_hash(raw_path) if recovery else None,
            "drawings_sha256": file_hash(drawings_path) if recovery else None,
        },
        "max_parent_chars_review_threshold": max_parent_chars,
        "ready_for_indexing": False,
    }
    return ParentPreparation(meta, sections, parents, issues, summary)


def write_preparation(result: ParentPreparation, output: Path) -> dict:
    """Write a new experiment, with summary.json as the completion marker."""
    sections, parents, issues, summary = (
        result.sections,
        result.parents,
        result.issues,
        result.summary,
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [
        ("sections.jsonl", sections),
        ("parents.jsonl", parents),
        ("issues.jsonl", issues),
    ]:
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output / "parents_preview.md").write_text(
        preview_markdown(summary, parents), encoding="utf-8"
    )
    # Written last: an interrupted directory without summary.json is incomplete.
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def prepare(
    pdf: Path,
    parsed_dir: Path,
    output: Path,
    scope_rules: Path | None = None,
    max_parent_chars: int = 10000,
    structure_rules: Path | None = None,
) -> dict:
    """Preserve the offline CLI contract; existing outputs are immutable."""
    if output.exists():
        raise ValueError("Output already exists; choose a new experiment directory")
    result = organize(pdf, parsed_dir, scope_rules, max_parent_chars, structure_rules)
    return write_preparation(result, output)
