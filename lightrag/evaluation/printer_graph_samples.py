"""Select real, traceable printer fragments for a bounded extraction experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lightrag.document_structure.capabilities import CapabilityIndex
from lightrag.document_structure.children import source_refs
from lightrag.document_structure.parents import file_hash
from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.document_structure.table_notes import (
    ID_RE,
    TABLE_RE,
    evidence_text,
    with_table_notes,
)
from lightrag.document_structure.tables import json_rows, parent_footnotes

QUERIES = [
    {"id": "G01", "query": "L5298 的网络连接报告中出现 E-12，该查看哪些说明？"},
    {"id": "G02", "query": "L5298 显示屏上出现 E-12，该查看哪些说明？"},
    {"id": "G03", "query": "L5298 出现 E-12，怎么办？"},
    {"id": "G04", "query": "L3266 显示 W-11，应按哪个按钮恢复打印？"},
    {"id": "G05", "query": "L4263 显示 W-11，应按哪个按钮恢复打印？"},
    {"id": "G06", "query": "显示屏出现 W-11，应按哪个按钮？"},
    {"id": "G07", "query": "L3268 能使用 ADF 扫描吗？"},
    {"id": "G08", "query": "L3266 能接收传真吗？"},
]


def only(items: list[dict], label: str) -> dict:
    if len(items) != 1:
        raise ValueError(f"Expected one {label}, found {len(items)}")
    return items[0]


def unit(
    index: ParentChildIndex,
    parent: dict,
    key: str,
    text: str,
    refs: list[dict],
    projection: dict,
) -> dict:
    child_ids = [
        child["child_id"]
        for child in index.children.values()
        if child["parent_id"] == parent["parent_id"]
        and any(
            a["block_id"] == b["block_id"]
            and max(a["start"], b["start"]) < min(a["end"], b["end"])
            for a in refs
            for b in child["source_spans"]
        )
    ]
    if not child_ids:
        raise ValueError("Sample has no source child attribution")
    return {
        "id": key,
        "doc_id": parent["doc_id"],
        "document_name": parent["document_name"],
        "parent_id": parent["parent_id"],
        "child_ids": child_ids,
        "heading_path": parent["heading_path"],
        "domain": parent["feature_domain"],
        "scope": parent["scope"],
        "pages": parent["pages"],
        "source_spans": refs,
        "projection": projection,
        "input_text": text,
    }


def table_row(index: ParentChildIndex, parent: dict, code: str, key: str) -> dict:
    matches = []
    for match in TABLE_RE.finditer(parent["content"]):
        rows, spans, _, _ = json_rows(match[0])
        for row_index, row in enumerate(rows):
            if row and str(row[0]).strip() == code:
                matches.append((match, rows, spans, row_index))
    if len(matches) != 1:
        raise ValueError(f"Expected one source table row for {code}")
    match, rows, spans, row_index = matches[0]
    identifier = ID_RE.search(match[0].split(">", 1)[0])[1]
    selected = [0, row_index]
    refs = []
    for position in selected:
        row_span = spans[position]
        row_refs, _ = source_refs(
            parent, match.start() + row_span["start"], match.start() + row_span["end"]
        )
        refs.extend(row_refs)
    notes = [
        note for note in parent.get("table_notes", []) if note["table_id"] == identifier
    ]
    text = json.dumps([rows[i] for i in selected], ensure_ascii=False)
    inline_notes = parent_footnotes(parent["content"], text)
    for note in inline_notes:
        span = note["parent_span"]
        note["source_spans"], note["source_positions"] = source_refs(
            parent, span["start"], span["end"]
        )
        refs.extend(note["source_spans"])
        text += "\n\n" + note["content"]
    result = unit(
        index,
        parent,
        key,
        with_table_notes(text, notes),
        refs,
        {
            "kind": "json_table_rows",
            "table_id": identifier,
            "row_indices": selected,
            "inline_note_spans": [note["parent_span"] for note in inline_notes],
        },
    )
    result["table_notes"] = notes
    result["inline_notes"] = inline_notes
    return result


def load_samples(root: Path) -> tuple[list[dict], dict]:
    indexes = {key: ParentChildIndex.from_directory(root / key) for key in ("B", "C")}
    b, c = indexes["B"], indexes["C"]
    network = only(
        [
            p
            for p in b.parents.values()
            if p["feature_domain"] == "network" and p["heading_path"][-1] == "E-12"
        ],
        "network E-12 parent",
    )
    display_b = only(
        [
            p
            for p in b.parents.values()
            if p["heading_path"][-1] == "显示屏上显示错误代码"
        ],
        "B display parent",
    )
    display_c = only(
        [
            p
            for p in c.parents.values()
            if p["heading_path"][-1] == "显示屏上显示错误代码"
        ],
        "C display parent",
    )
    adf = only(
        [p for p in b.parents.values() if p["heading_path"][-1] == "ADF 规格"],
        "ADF specifications",
    )
    records = CapabilityIndex(list(b.parents.values())).records
    intro_ids = {record["parent_id"] for record in records}
    if len(intro_ids) != 1 or {r["feature"] for r in records} != {
        "adf",
        "fax",
        "ethernet",
    }:
        raise ValueError("Expected a verified exclusive feature list")
    intro = b.parents[next(iter(intro_ids))]
    start = min(r["parent_span"]["start"] for r in records)
    end = max(r["parent_span"]["end"] for r in records)
    refs, _ = source_refs(intro, start, end)
    samples = [
        unit(
            b,
            network,
            "S01_network_E12",
            evidence_text(network),
            network["source_spans"],
            {"kind": "whole_parent"},
        ),
        table_row(b, display_b, "E-12", "S02_display_E12"),
        table_row(b, display_b, "W-11", "S03_B_W11"),
        table_row(c, display_c, "W-11", "S04_C_W11"),
        unit(
            b,
            intro,
            "S05_feature_limits",
            intro["content"][start:end],
            refs,
            {"kind": "parent_substring", "start": start, "end": end},
        ),
        unit(
            b,
            adf,
            "S06_ADF_specs",
            evidence_text(adf),
            adf["source_spans"],
            {"kind": "whole_parent"},
        ),
    ]
    return samples, {key: file_hash(root / key / "summary.json") for key in indexes}


def query_slots(query: str) -> dict:
    from lightrag.document_structure.capabilities import query_features
    from lightrag.document_structure.parents import MODEL_RE

    domain = (
        "network"
        if re.search(r"网络(?:连接)?报告", query)
        else "device_display_errors"
        if re.search(r"显示屏|屏幕", query)
        else None
    )
    return {
        "models": sorted(set(MODEL_RE.findall(query.upper()))),
        "codes": sorted(set(re.findall(r"\b[A-Z]{1,3}-\d{1,4}\b", query.upper()))),
        "domain": domain,
        "features": query_features(query),
    }
