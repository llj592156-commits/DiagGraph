"""Parent evidence lookup preserves sources and never modifies graph candidates."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from lightrag.document_structure.query import expand_parent_evidence

pytestmark = pytest.mark.offline


def stores():
    rows = {
        "doc-B-pc-child-one": {
            "full_doc_id": "doc-B",
            "parent_child": {
                "doc_id": "doc-B",
                "parent_id": "p1",
                "context_parent_ids": ["p0"],
            },
        },
    }
    rows["doc-B-pc-child-two"] = deepcopy(rows["doc-B-pc-child-one"])
    bundle = {
        "doc_id": "doc-B",
        "parents": {
            "p1": {
                "doc_id": "doc-B",
                "content": "Full parent",
                "pages": [2],
                "table_notes": [{"table_id": "tb-1", "content": "Only for L5298"}],
            },
            "p0": {"doc_id": "doc-B", "content": "Prerequisite parent", "pages": [1]},
        },
    }
    chunks, docs = AsyncMock(), AsyncMock()
    chunks.get_by_ids.side_effect = lambda ids: [rows.get(key) for key in ids]
    docs.get_by_ids.return_value = [{"parent_child": bundle}]
    return chunks, docs, rows, bundle


@pytest.mark.asyncio
async def test_parent_dedup_notes_prerequisites_and_ordinary_chunks():
    chunks, docs, rows, _ = stores()
    ordinary = {"chunk_id": "ordinary", "content": "plain text"}
    hits = [
        {"chunk_id": key, "content": "child", "file_path": "guide.pdf"} for key in rows
    ]
    expanded, info = await expand_parent_evidence([ordinary, *hits], chunks, docs)
    assert len(expanded) == 2
    assert expanded[0] is ordinary
    assert expanded[1]["parent_ids"] == ["p1", "p0"]
    assert all(
        s in expanded[1]["content"]
        for s in ["Full parent", "Only for L5298", "Prerequisite parent"]
    )
    assert [p["pages"] for p in info["parents"]] == [[2], [1]]
    assert info["parents"][0]["source_chunk_id"] == hits[0]["chunk_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["chunk", "document", "parent", "owner"])
async def test_missing_attribution_fails_loud(failure):
    chunks, docs, rows, bundle = stores()
    if failure == "chunk":
        rows.clear()
    elif failure == "document":
        docs.get_by_ids.return_value = [None]
    elif failure == "parent":
        bundle["parents"].pop("p1")
    else:
        bundle["parents"]["p1"]["doc_id"] = "other-owner"
    with pytest.raises(ValueError, match="missing|different owner"):
        await expand_parent_evidence([{"chunk_id": "doc-B-pc-child-one"}], chunks, docs)


@pytest.mark.asyncio
async def test_ordinary_chunks_skip_extra_storage_reads():
    chunks, docs = AsyncMock(), AsyncMock()
    hits = [{"chunk_id": "ordinary", "content": "plain text"}]
    result, info = await expand_parent_evidence(hits, chunks, docs)
    assert result is hits and info is None
    chunks.get_by_ids.assert_not_called()
    docs.get_by_ids.assert_not_called()
