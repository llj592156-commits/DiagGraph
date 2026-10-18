"""H strategy: prepare parents and indexable children inside the existing pipeline.

Parents belong to the owning full_docs row, children to text_chunks. No extra
store is created. See docs/ParentChildSystemIntegration-zh.md for lifecycle rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lightrag.document_structure.children import ChildConfig, build_children
from lightrag.document_structure.parents import file_hash, organize
from lightrag.utils import Tokenizer

VERSION = "parent-child-ingestion-v1"


@dataclass
class ParentChildPreparation:
    chunks: list[dict]
    bundle: dict


def child_evidence_content(child: dict) -> str:
    """Keep the complete saved child and its required supplemental notes."""
    content = child["content"]
    for note in [*child.get("required_notes", []), *child.get("table_notes", [])]:
        if note["content"] not in content:
            content += "\n\n" + note["content"]
    return content


def prepare_parent_child(
    tokenizer: Tokenizer,
    pdf: str,
    blocks_path: str,
    doc_id: str,
    document_name: str,
    *,
    chunk_token_size: int = 500,
    chunk_overlap_token_size: int = 60,
) -> ParentChildPreparation:
    """Require the original PDF and structured sidecars; never fall back to F."""
    path = Path(blocks_path)
    rules = Path(__file__).resolve().parents[2] / "examples/parent_blocks"
    scope_rules = rules / "printer_scope_rules.json"
    structure_rules = rules / "printer_structure_rules.json"
    header_rules = rules / "printer_table_headers.json"
    if not all(p.is_file() for p in (scope_rules, structure_rules, header_rules)):
        raise ValueError(
            "H requires the repository's versioned printer structure rules"
        )
    result = organize(
        Path(pdf),
        path.parent,
        scope_rules=scope_rules,
        structure_rules=structure_rules,
        document_name=document_name,
        doc_id=doc_id,
    )
    tables_path = path.with_name(path.name.replace(".blocks.jsonl", ".tables.json"))
    tables, headers = None, {}
    if tables_path.is_file():
        tables = json.loads(tables_path.read_text(encoding="utf-8"))["tables"]
        for key, table in tables.items():
            if table.get("format") == "json" and table.get("table_header"):
                headers[key] = json.loads(table["table_header"])
    for key, value in json.loads(header_rules.read_text(encoding="utf-8"))[
        "tables"
    ].items():
        if value["evidence"]["document_name"] == document_name:
            headers[key] = value["header_rows"]
    parents, children = build_children(
        result.parents,
        tokenizer,
        ChildConfig(chunk_token_size, chunk_overlap_token_size),
        table_headers=headers,
        table_sidecar=tables,
        table_sidecar_sha256=file_hash(tables_path) if tables is not None else None,
    )
    chunks = []
    for index, child in enumerate(children):
        content = child_evidence_content(child)
        metadata = {
            key: child[key]
            for key in (
                "parent_id",
                "child_id",
                "doc_id",
                "document_name",
                "scope",
                "feature_domain",
                "conditions",
                "context_parent_ids",
                "parent_span",
                "source_spans",
                "source_positions",
                "heading_path",
                "pages",
            )
        }
        metadata["version"] = VERSION
        chunks.append(
            {
                "chunk_id": "pc-" + child["child_id"],
                "content": content,
                "tokens": len(tokenizer.encode(content)),
                "chunk_order_index": index,
                "heading": {
                    "heading": child["heading_path"][-1]
                    if child["heading_path"]
                    else "",
                    "parent_headings": child["heading_path"][:-1],
                },
                "parent_child": metadata,
            }
        )
    bundle = {
        "version": VERSION,
        "doc_id": doc_id,
        "parents": {p["parent_id"]: p for p in parents},
        "issues": result.issues,
        "source": result.summary["inputs"],
        "rules": {
            p.name: file_hash(p) for p in (scope_rules, structure_rules, header_rules)
        },
        "child_count": len(chunks),
    }
    return ParentChildPreparation(chunks, bundle)
