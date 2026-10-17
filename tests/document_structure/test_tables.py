"""Table projections preserve every data row, repeated headers and notes."""

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
from lightrag.document_structure.tables import (
    json_rows,
    slice_json_table,
    validate_table_part,
)

from .test_children import parent

pytestmark = pytest.mark.offline


@pytest.fixture
def tokenizer():
    return SimpleNamespace(
        model_name="characters-test-only", encode=lambda text: list(map(ord, text))
    )


def table_text():
    rows = [["Code", "Solution"]] + [
        [f"E-{i}", "Keep this complete row *1 " + "x" * 25] for i in range(8)
    ]
    return '<table id="table" format="json">' + json.dumps(rows) + "</table>"


def test_row_slices_retain_source_values_and_offsets(tokenizer):
    table = table_text()
    rows, spans, _, _ = json_rows(table)
    parts = slice_json_table(table, rows[:1], tokenizer, 180)
    assert len(parts) > 1
    combined = []
    for part in parts:
        validate_table_part(table, part)
        sliced, _, _, _ = json_rows(part["content"])
        assert sliced[0] == rows[0]
        combined.extend(sliced[1:])
        assert len(tokenizer.encode(part["content"])) <= 180
        for span in part["transformation"]["row_spans"]:
            assert (
                json.loads(table[span["start"] : span["end"]])
                == rows[span["row_index"]]
            )
    assert combined == rows[1:]
    assert len(spans) == len(rows)


def test_foreign_header_and_row_edits_are_rejected(tokenizer):
    table = table_text()
    with pytest.raises(ValueError, match="header"):
        slice_json_table(table, [["wrong"]], tokenizer, 180)
    part = slice_json_table(table, [["Code", "Solution"]], tokenizer, 180)[0]
    part["content"] = part["content"].replace("E-0", "E-99")
    with pytest.raises(ValueError, match="altered source rows"):
        validate_table_part(table, part)


def test_children_keep_notes_and_return_the_complete_parent(tokenizer):
    content = (
        table_text() + "\n*1 Required warning, do not discard.\n\n## Next\nInstructions"
    )
    original = parent(text=content)
    config = ChildConfig(180, 10)
    parents, children = build_children(
        [original], tokenizer, config, table_headers={"table": [["Code", "Solution"]]}
    )
    projected = [c for c in children if c.get("transformation")]
    assert len(projected) > 1
    assert all(
        c["required_notes"][0]["content"] == "*1 Required warning, do not discard."
        for c in projected
    )
    assert all("Required warning" in c["retrieval_text"] for c in projected)
    assert all(c["source_spans"] for c in projected)
    result = ParentChildIndex(parents, children).resolve(
        [projected[0]["child_id"]], model="L3266"
    )
    assert result["parents"][0]["content"] == content
    checks = validate_children(parents, children, tokenizer, config)
    assert checks["table_rows_verified"]
    assert not checks["exact_parent_substrings"]


@pytest.mark.parametrize(
    "damage", ["drop", "duplicate", "header", "footnote", "source"]
)
def test_projection_validation_rejects_structural_damage(tokenizer, damage):
    content = table_text() + "\n*1 Required warning."
    config = ChildConfig(180, 10)
    parents, children = build_children(
        [parent(text=content)],
        tokenizer,
        config,
        table_headers={"table": [["Code", "Solution"]]},
    )
    first = next(c for c in children if c.get("transformation"))
    if damage == "drop":
        children.remove(first)
        for i, c in enumerate(children):
            c["order_in_parent"] = i
    elif damage == "duplicate":
        extra = deepcopy(first)
        extra["child_id"] += "-duplicate"
        children.insert(1, extra)
        for i, c in enumerate(children):
            c["order_in_parent"] = i
    elif damage == "header":
        first["content"] = first["content"].replace("Solution", "Incorrect")
    elif damage == "footnote":
        first["required_notes"] = []
    else:
        first["source_spans"][0]["start"] += 1
    with pytest.raises(ValueError):
        validate_children(parents, children, tokenizer, config)


def test_single_long_row_is_retained_and_flagged(tokenizer):
    rows = [["H"], ["x" * 300], ["y"]]
    table = '<table id="table" format="json">' + json.dumps(rows) + "</table>"
    _, children = build_children(
        [parent(text=table)],
        tokenizer,
        ChildConfig(100, 5),
        table_headers={"table": rows[:1]},
    )
    assert any("oversized_table_row" in c["review_flags"] for c in children)
    assert "x" * 300 in children[0]["content"]
