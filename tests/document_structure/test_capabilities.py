"""Explicit feature declarations are evidence, independent of parent scope."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from lightrag.document_structure.capabilities import (
    CapabilityIndex,
    extract_capabilities,
    query_features,
)
from lightrag.document_structure.children import build_children
from lightrag.document_structure.retrieval import ParentChildIndex

from .test_children import parent

pytestmark = pytest.mark.offline


def declaration(models="L5298", key="intro", feature="ADF（自动文稿送纸器）"):
    return parent(
        key,
        f"## Introduction\n这是共用手册。以下功能仅适用于 {models}。\nO {feature}\nO 传真\nNext section\nO 以太网连接",
        models=["L3266", "L5298"],
    )


def test_exclusive_list_preserves_quotes_offsets_and_filename_scope():
    p = declaration()
    before = deepcopy(p)
    records = extract_capabilities([p])
    assert p == before
    assert {r["feature"] for r in records} == {"adf", "fax"}
    for r in records:
        start, end = r["parent_span"].values()
        assert p["content"][start:end] == r["content"]
        assert r["source_spans"][0]["start"] == 10 + start
        assert r["supported_models"] == ["L5298"]
    idx = CapabilityIndex([p])
    assert idx.check("l5298", "adf")["status"] == "supported"
    assert idx.check("L3266", "adf")["status"] == "unsupported"
    assert idx.check("L4263", "adf")["status"] == "unknown"
    assert idx.check("L3266", "ethernet")["status"] == "unknown"


@pytest.mark.parametrize(
    "text",
    [
        "ADF 使用说明",
        "以下功能适用于 L5298。\nO 传真",
        "以下功能仅适用于 L9999。\nO 传真",
        "以下功能仅适用于 L5298。\nSome prose\nO 传真",
    ],
)
def test_features_and_unrecognized_wording_do_not_infer_exclusivity(text):
    assert extract_capabilities([parent(text=text)]) == []


def test_conflicting_declarations_fail_closed():
    idx = CapabilityIndex([declaration(), declaration("L3266", "other")])
    check = idx.check("L3266", "fax")
    assert check["status"] == "conflict" and check["requires_review"]
    assert len(check["evidence"]) == 2


def test_unsupported_operations_are_rejected_but_explanation_is_retained():
    tokenizer = SimpleNamespace(model_name="characters", encode=lambda text: list(text))
    parents, children = build_children(
        [declaration(), parent("operation", "ADF loading steps")], tokenizer
    )
    idx = ParentChildIndex(parents, children)
    child = next(c for c in children if c["parent_id"] == "operation")
    # Existing filename-default policy remains intact when no feature is requested.
    assert idx.resolve([child["child_id"]], model="L3266")["parents"]
    rejected = idx.resolve(
        [child["child_id"]], model="L3266", required_features=["adf"]
    )
    assert rejected["parents"] == []
    assert rejected["rejected"][0]["reason"] == "feature_capability_not_supported"
    assert "仅适用于 L5298" in rejected["capability_evidence"][0]["content"]
    assert idx.resolve([child["child_id"]], model="L5298", required_features=["adf"])[
        "parents"
    ]
    assert parents[1]["scope"]["applicable_models"] == ["L3266", "L5298"]


def test_unknown_support_is_flagged_instead_of_claimed():
    tokenizer = SimpleNamespace(model_name="characters", encode=lambda text: list(text))
    parents, children = build_children([parent()], tokenizer)
    result = ParentChildIndex(parents, children).resolve(
        [children[0]["child_id"]], model="L3266", required_features=["adf"]
    )
    assert result["capability_checks"][0]["status"] == "unknown"
    assert result["requires_review"]


def test_query_feature_detection_is_literal_and_checks_every_model():
    idx = CapabilityIndex([declaration()])
    assert query_features("BIOS printing") == []
    checks = idx.check_query("L3266 和 L5298 能用 ADF 吗？")
    assert [(c["model"], c["status"]) for c in checks] == [
        ("L3266", "unsupported"),
        ("L5298", "supported"),
    ]
