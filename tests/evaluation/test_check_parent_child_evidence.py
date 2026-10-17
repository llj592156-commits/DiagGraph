"""The audit distinguishes source-body coverage from supplemental evidence."""

from types import SimpleNamespace

import pytest

from lightrag.document_structure.children import build_children
from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.evaluation.check_parent_child_evidence import audit
from tests.document_structure.test_table_notes import HASH, fixture_data

pytestmark = pytest.mark.offline


def test_inventory_checks_notes_without_embedding_or_answer_generation():
    p, tables = fixture_data()
    parents, children = build_children(
        [p],
        SimpleNamespace(model_name="chars", encode=lambda text: list(text)),
        table_sidecar=tables,
        table_sidecar_sha256=HASH,
    )
    result = audit(
        {"C": ParentChildIndex(parents, children)},
        [
            {
                "id": "Q",
                "query": "Pad lifetime",
                "evidence": [
                    {"document": "C", "quote": "Pad lifetime is not printer lifetime"},
                ],
            }
        ],
    )
    assert result["embedding_requests"] == result["generation_model_calls"] == 0
    anchor = result["anchor_inventory"][0]["anchors"][0]
    assert not anchor["source_body_contains"] and anchor["evidence_contains"]
