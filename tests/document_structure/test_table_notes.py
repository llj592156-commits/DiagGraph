"""Sidecar footnotes survive indexing and evidence expansion with real provenance."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lightrag.document_structure.children import (
    ChildConfig,
    build_children,
    validate_children,
)
from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.document_structure.table_notes import evidence_text, notes_for_parent

from .test_children import parent

pytestmark = pytest.mark.offline
HASH = "sha256:" + "a" * 64


def fixture_data():
    rows = [["Code", "Solution"], ["W-11", "Replace pad *"]]
    body = json.dumps(rows)
    text = f'<table id="tb" format="json">{body}</table>'
    tables = {
        "tb": {
            "format": "json",
            "content": body,
            "footnotes": ["* Pad lifetime is not printer lifetime."],
        }
    }
    return parent(text=text), tables


def build(config=None):
    p, tables = fixture_data()
    tokenizer = SimpleNamespace(model_name="characters", encode=lambda text: list(text))
    config = config or ChildConfig()
    parents, children = build_children(
        [p],
        tokenizer,
        config,
        table_sidecar=tables,
        table_sidecar_sha256=HASH,
        table_headers={"tb": [["Code", "Solution"]]},
    )
    return parents, children, tokenizer, config


@pytest.mark.parametrize("target", [500, 80])
def test_unnumbered_notes_survive_small_and_projected_tables(target):
    p, tables = fixture_data()
    before = deepcopy(p)
    parents, children, tokenizer, config = build(ChildConfig(target, 5))
    assert p == before
    table_child = next(c for c in children if c["assets"])
    note = table_child["table_notes"][0]
    assert note["source"]["json_pointer"] == "/tables/tb/footnotes/0"
    assert note["source"]["sha256"] == HASH
    assert "parent_span" not in note and "source_spans" not in note
    assert parents[0]["content"] == p["content"]
    assert note["content"] in table_child["retrieval_text"]
    evidence = ParentChildIndex(parents, children).resolve(
        [table_child["child_id"]], model="L3266"
    )
    assert note["content"] in evidence["parents"][0]["evidence_content"]
    assert evidence["parents"][0]["content"] == p["content"]
    assert (
        validate_children(parents, children, tokenizer, config)["table_note_count"] == 1
    )
    assert parents[0]["table_notes"] == notes_for_parent(p, tables, HASH)


@pytest.mark.parametrize(
    "damage", ["drop", "rewrite", "pointer", "table", "hash", "embedding"]
)
def test_notes_cannot_be_silently_lost_or_misattributed(damage):
    parents, children, tokenizer, config = build()
    if damage == "drop":
        children[0]["table_notes"] = []
    elif damage == "embedding":
        children[0]["retrieval_text"] = children[0]["content"]
        children[0]["retrieval_tokens"] = len(children[0]["content"])
    elif damage == "rewrite":
        children[0]["table_notes"][0]["content"] = "Changed warning"
    else:
        note = parents[0]["table_notes"][0]
        if damage == "pointer":
            note["source"]["json_pointer"] = "/tables/foreign/footnotes/0"
        elif damage == "table":
            note["table_id"] = "foreign"
        else:
            note["source"]["sha256"] = "invalid"
    with pytest.raises(ValueError):
        validate_children(parents, children, tokenizer, config)


def test_foreign_rows_and_missing_source_tables_are_refused():
    p, tables = fixture_data()
    with pytest.raises(ValueError, match="absent"):
        notes_for_parent(p, {}, HASH)
    tables["tb"]["content"] = '[["Changed"]]'
    with pytest.raises(ValueError, match="differs"):
        notes_for_parent(p, tables, HASH)


def test_notes_are_attached_only_to_their_table_children():
    p, tables = fixture_data()
    p = parent(text="Unrelated prose.\n" + p["content"])
    tokenizer = SimpleNamespace(model_name="characters", encode=lambda text: list(text))
    parents, children = build_children(
        [p],
        tokenizer,
        ChildConfig(80, 5),
        table_sidecar=tables,
        table_sidecar_sha256=HASH,
    )
    prose = next(c for c in children if c["kind"] == "text")
    assert prose["table_notes"] == []
    assert "Pad lifetime" not in prose["retrieval_text"]
    assert "Pad lifetime" in evidence_text(parents[0])


def test_notes_already_in_source_content_are_not_duplicated():
    parents, _, _, _ = build()
    note = parents[0]["table_notes"][0]["content"]
    parents[0]["content"] += "\n" + note
    assert evidence_text(parents[0]).count(note) == 1


@pytest.mark.parametrize("damage", ["source", "consistent_removal"])
def test_bundle_load_rechecks_notes_against_hashed_source(tmp_path, damage):
    from lightrag.document_structure.children import VERSION
    from lightrag.document_structure.parents import file_hash

    p, tables = fixture_data()
    source = tmp_path / "tables.json"
    source.write_text(json.dumps({"tables": tables}), encoding="utf-8")
    parents, children = build_children(
        [p],
        SimpleNamespace(model_name="chars", encode=lambda text: list(text)),
        table_sidecar=tables,
        table_sidecar_sha256=file_hash(source),
    )
    if damage == "consistent_removal":
        parents[0]["table_notes"] = []
        children[0]["table_notes"] = []
    for name, rows in [("parents.jsonl", parents), ("children.jsonl", children)]:
        (tmp_path / name).write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    summary = {
        "schema_version": VERSION,
        "parents": len(parents),
        "children": len(children),
        "doc_id": p["doc_id"],
        "document_name": p["document_name"],
        "outputs": {
            name: file_hash(tmp_path / name)
            for name in ("parents.jsonl", "children.jsonl")
        },
        "inputs": {
            "table_sidecar": str(source),
            "table_sidecar_sha256": file_hash(source),
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if damage == "source":
        source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        ParentChildIndex.from_directory(tmp_path)
