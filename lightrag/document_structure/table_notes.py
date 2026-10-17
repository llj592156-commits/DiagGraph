"""Keep table-sidecar notes separate from immutable source-block offsets."""

from __future__ import annotations

import json
import re
from copy import deepcopy

from .tables import json_rows

TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL)
ID_RE = re.compile(r'\bid="([^"]+)"')


def notes_for_parent(parent: dict, tables: dict, source_hash: str) -> list[dict]:
    """Attach only notes belonging to unchanged tables present in this parent.

    Source references are table-sidecar JSON pointers, never fabricated block
    character ranges. The caller must record and verify the sidecar hash.
    """
    notes = []
    seen = set()
    for match in TABLE_RE.finditer(parent["content"]):
        identifier = ID_RE.search(match[0].split(">", 1)[0])
        if not identifier:
            raise ValueError("Table has no source ID")
        key = identifier[1]
        if key in seen:
            continue
        seen.add(key)
        table = tables.get(key)
        if table is None:
            raise ValueError("Parent table is absent from table sidecar")
        if table.get("format") == "json" and json_rows(match[0])[0] != json.loads(
            table["content"]
        ):
            raise ValueError("Parent table differs from table sidecar")
        footnotes = table.get("footnotes", [])
        if not isinstance(footnotes, list) or any(
            not isinstance(note, str) or not note.strip() for note in footnotes
        ):
            raise ValueError("Invalid table sidecar footnotes")
        escaped = key.replace("~", "~0").replace("/", "~1")
        for index, content in enumerate(footnotes):
            notes.append(
                {
                    "table_id": key,
                    "content": content,
                    "source": {
                        "kind": "table_sidecar",
                        "sha256": source_hash,
                        "json_pointer": f"/tables/{escaped}/footnotes/{index}",
                        "doc_id": parent["doc_id"],
                        "document_name": parent["document_name"],
                    },
                }
            )
    return notes


def notes_for_child(parent: dict, child: dict) -> list[dict]:
    identifiers = {
        item["id"] for item in child.get("assets", []) if item["kind"] == "table"
    }
    # Retain every note of a referenced table: unnumbered '*' notes and notes
    # spanning several rows cannot safely be assigned by a numeric-marker regex.
    return deepcopy(
        [
            note
            for note in parent.get("table_notes", [])
            if note["table_id"] in identifiers
        ]
    )


def validate_notes(parent: dict) -> None:
    identifiers = {
        ID_RE.search(match[0].split(">", 1)[0])[1]
        for match in TABLE_RE.finditer(parent["content"])
        if ID_RE.search(match[0].split(">", 1)[0])
    }
    pointers = set()
    for note in parent.get("table_notes", []):
        source = note["source"]
        key = note["table_id"]
        escaped = key.replace("~", "~0").replace("/", "~1")
        pointer = source["json_pointer"]
        if (
            key not in identifiers
            or source["kind"] != "table_sidecar"
            or source["doc_id"] != parent["doc_id"]
            or source["document_name"] != parent["document_name"]
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", source["sha256"])
            or not re.fullmatch(
                re.escape(f"/tables/{escaped}/footnotes/") + r"\d+", pointer
            )
            or pointer in pointers
            or not isinstance(note["content"], str)
            or not note["content"].strip()
        ):
            raise ValueError("Invalid table note attribution")
        pointers.add(pointer)


def with_table_notes(content: str, notes: list[dict]) -> str:
    """Render supplemental evidence without altering source-block content."""
    result = content

    def normalized(value: str) -> str:
        return re.sub(r"\s+", "", value)

    for note in notes:
        if normalized(note["content"]) not in normalized(result):
            result += f"\n\nTable note ({note['table_id']}):\n{note['content']}"
    return result


def evidence_text(parent: dict) -> str:
    return with_table_notes(parent["content"], parent.get("table_notes", []))
