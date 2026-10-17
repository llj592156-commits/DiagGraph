"""Row-preserving JSON table slices with verifiable source intervals.

Only explicit, prefix-matching headers may repeat. A single oversized row is
retained and flagged by the caller. No row, cell, or footnote is truncated.
"""

from __future__ import annotations

import json
import re

from lightrag.table_markup import TABLE_TAG_RE, extract_table_id
from lightrag.utils import Tokenizer


def json_rows(table: str) -> tuple[list, list[dict], int, int]:
    match = TABLE_TAG_RE.fullmatch(table)
    if match is None or not re.search(r'format=["\']json["\']', match["attrs"]):
        raise ValueError("Row slicing requires one JSON table")
    body = match["body"]
    rows = json.loads(body)
    if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
        raise ValueError("JSON table must contain row arrays")
    decoder, spans = json.JSONDecoder(), []
    cursor = body.index("[") + 1
    for index, row in enumerate(rows):
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        value, end = decoder.raw_decode(body, cursor)
        if value != row:
            raise ValueError("JSON row span mismatch")
        spans.append(
            {
                "row_index": index,
                "start": match.start("body") + cursor,
                "end": match.start("body") + end,
            }
        )
        cursor = end
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if index + 1 < len(rows):
            if body[cursor] != ",":
                raise ValueError("Missing JSON row separator")
            cursor += 1
    return rows, spans, match.start("body"), match.end("body")


def slice_json_table(
    table: str, header: list, tokenizer: Tokenizer, target: int
) -> list[dict]:
    rows, spans, body_start, body_end = json_rows(table)
    count = len(header)
    if not count or rows[:count] != header:
        raise ValueError("Explicit table header is not a source row prefix")
    if len(rows) <= count:
        return []
    identifier = extract_table_id(TABLE_TAG_RE.fullmatch(table)["attrs"])

    def part(start: int, end: int) -> dict:
        selected = [*range(count), *range(start, end)]
        content = (
            table[:body_start]
            + "["
            + ", ".join(table[spans[i]["start"] : spans[i]["end"]] for i in selected)
            + "]"
            + table[body_end:]
        )
        return {
            "content": content,
            "transformation": {
                "kind": "json_table_rows",
                "table_id": identifier,
                "header_count": count,
                "row_start": start,
                "row_end": end,
                "total_rows": len(rows),
                "row_spans": [spans[i] for i in selected],
                "wrapper_spans": [
                    {"start": 0, "end": body_start},
                    {"start": body_end, "end": len(table)},
                ],
            },
        }

    result, start = [], count
    for end in range(count + 1, len(rows) + 1):
        candidate = part(start, end)
        if len(tokenizer.encode(candidate["content"])) > target and end > start + 1:
            result.append(part(start, end - 1))
            start = end - 1
    result.append(part(start, len(rows)))
    return result


def validate_table_part(table: str, part: dict) -> None:
    rows, spans, body_start, body_end = json_rows(table)
    meta = part["transformation"]
    if meta["kind"] != "json_table_rows":
        raise ValueError("Unsupported table projection")
    count, start, end = meta["header_count"], meta["row_start"], meta["row_end"]
    if not 1 <= count <= start < end <= len(rows) or meta["total_rows"] != len(rows):
        raise ValueError("Invalid table row interval")
    selected = [*range(count), *range(start, end)]
    expected = (
        table[:body_start]
        + "["
        + ", ".join(table[spans[i]["start"] : spans[i]["end"]] for i in selected)
        + "]"
        + table[body_end:]
    )
    if part["content"] != expected or meta["row_spans"] != [spans[i] for i in selected]:
        raise ValueError("Table slice altered source rows or header")
    if meta["wrapper_spans"] != [
        {"start": 0, "end": body_start},
        {"start": body_end, "end": len(table)},
    ]:
        raise ValueError("Table slice wrapper source mismatch")
    if meta["table_id"] != extract_table_id(TABLE_TAG_RE.fullmatch(table)["attrs"]):
        raise ValueError("Table slice ID mismatch")


def parent_footnotes(content: str, selected_text: str) -> list[dict]:
    needed = set(re.findall(r"\*\s*(\d+)", selected_text))
    starts = list(re.finditer(r"(?m)^\*(\d+)\s+", content))
    notes = []
    for index, match in enumerate(starts):
        if match[1] not in needed:
            continue
        end = starts[index + 1].start() if index + 1 < len(starts) else len(content)
        stop = re.search(r"\n\s*(?:##|<table\b|<drawing\b)", content[match.end() : end])
        if stop:
            end = match.end() + stop.start()
        while end > match.start() and content[end - 1].isspace():
            end -= 1
        notes.append(
            {
                "kind": "footnote",
                "marker": match[1],
                "parent_span": {"start": match.start(), "end": end},
                "content": content[match.start() : end],
            }
        )
    return notes
