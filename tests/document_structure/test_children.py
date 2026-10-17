"""Offline child drafts preserve source ranges, markup and applicability."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from lightrag.document_structure import children as childmod
from lightrag.document_structure.retrieval import ParentChildIndex

pytestmark = pytest.mark.offline


@pytest.fixture
def tokenizer():
    return SimpleNamespace(
        model_name="characters-test-only", encode=lambda text: list(map(ord, text))
    )


def parent(
    key="p",
    text="Instructions",
    path=None,
    models=None,
    contexts=None,
    domain="unknown",
):
    return {
        "parent_id": key,
        "doc_id": "doc-one",
        "document_name": "L3266_L5298.pdf",
        "heading_path": path or ["Guide"],
        "content": text,
        "content_role": "body",
        "scope": {
            "status": "filename",
            "applicable_models": models or ["L3266", "L5298"],
        },
        "context_parent_ids": contexts or [],
        "feature_domain": domain,
        "review_flags": [],
        "source_spans": [
            {
                "block_id": key + "-source",
                "fragment_id": key + "-fragment",
                "start": 10,
                "end": 10 + len(text),
            }
        ],
        "source_positions": [
            {
                "block_id": key + "-source",
                "fragment_id": key + "-fragment",
                "positions": [
                    {"type": "bbox", "anchor": "2", "range": [0, 20, 30, 10]}
                ],
            }
        ],
    }


def test_repeated_text_offsets_are_exact_and_inputs_immutable(tokenizer):
    parents = [parent(text=("重复步骤。\n" * 60))]
    before = deepcopy(parents)
    config = childmod.ChildConfig(50, 8)
    annotated, children = childmod.build_children(parents, tokenizer, config)
    assert parents == before
    assert len(children) > 1
    assert all(c["tokens"] <= 50 for c in children)
    for c in children:
        span = c["parent_span"]
        assert c["content"] == parents[0]["content"][span["start"] : span["end"]]
        assert c["source_spans"][0]["start"] == 10 + span["start"]
        assert c["source_spans"][0]["end"] == 10 + span["end"]
    assert childmod.validate_children(annotated, children, tokenizer, config)[
        "nonwhitespace_content_covered"
    ]
    assert (annotated, children) == childmod.build_children(parents, tokenizer, config)


def test_table_and_drawing_are_preserved_including_oversize(tokenizer):
    table = '<table id="tb" format="json">[["header","' + "value" * 30 + '"]]</table>'
    drawing = '<drawing id="im" format="png" path="image.png" />'
    text = (
        "Warning: keep context.\n" + table + "\nNext step.\n" + drawing + "\nFootnote."
    )
    annotated, children = childmod.build_children(
        [parent(text=text)], tokenizer, childmod.ChildConfig(70, 10)
    )
    table_child = next(c for c in children if c["kind"] == "table")
    assert table_child["content"] == table
    assert table_child["assets"] == [{"kind": "table", "id": "tb"}]
    assert "oversized_atomic_asset" in table_child["review_flags"]
    assert next(c for c in children if c["kind"] == "drawing")["content"] == drawing
    assert all(not c["ready_for_indexing"] for c in children)
    assert childmod.validate_children(
        annotated, children, tokenizer, childmod.ChildConfig(70, 10)
    )["protected_markup_preserved"]
    result = ParentChildIndex(annotated, children).resolve(
        [table_child["child_id"]], model="L5298"
    )
    assert result["requires_review"]
    assert "oversized_atomic_asset" in result["review_flags"]


def test_nonbody_parents_have_no_children(tokenizer):
    parents = [parent("a"), {**parent("toc"), "content_role": "table_of_contents"}]
    _, children = childmod.build_children(parents, tokenizer)
    assert [c["parent_id"] for c in children] == ["a"]


def test_source_ranges_map_multiple_blocks_without_using_item_charspan(tokenizer):
    p = parent(text="ABC\n\nDEF")
    p["source_spans"] = [
        {"block_id": "original", "fragment_id": "f1", "start": 5, "end": 8},
        {"block_id": "other", "fragment_id": "f2", "start": 12, "end": 15},
    ]
    p["source_positions"] = [
        {
            "block_id": "original",
            "fragment_id": "f1",
            "positions": [{"anchor": "1", "charspan": [0, 3]}],
        },
        {
            "block_id": "other",
            "fragment_id": "f2",
            "positions": [{"anchor": "2", "charspan": [0, 3]}],
        },
    ]
    _, children = childmod.build_children([p], tokenizer)
    assert children[0]["source_spans"] == p["source_spans"]
    assert children[0]["pages"] == [1, 2]
    assert children[0]["position_precision"] == "source_fragment"


@pytest.mark.parametrize(
    "defect", ["drop", "duplicate", "rewrite", "scope", "source", "tokens", "order"]
)
def test_child_validation_rejects_loss_and_metadata_damage(tokenizer, defect):
    config = childmod.ChildConfig(40, 5)
    parents, children = childmod.build_children(
        [parent(text="Step.\n" * 40)], tokenizer, config
    )
    if defect == "drop":
        children.pop()
    elif defect == "duplicate":
        children.append(deepcopy(children[0]))
    elif defect == "rewrite":
        children[0]["content"] = "Altered"
    elif defect == "scope":
        children[0]["scope"]["applicable_models"] = ["L9999"]
    elif defect == "source":
        children[0]["source_spans"][0]["start"] += 1
    elif defect == "tokens":
        children[0]["tokens"] += 1
    else:
        children.reverse()
    with pytest.raises(ValueError):
        childmod.validate_children(parents, children, tokenizer, config)


def test_unclosed_table_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="Incomplete protected markup"):
        childmod.build_children([parent(text='<table id="broken">Row')], tokenizer)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_child_parameters_fail_early(size, overlap):
    with pytest.raises(ValueError):
        childmod.ChildConfig(size, overlap)


def test_local_tokenizer_rejects_wrong_vocabulary(tmp_path):
    path = tmp_path / "bad.tiktoken"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="SHA-256"):
        childmod.load_local_tokenizer(path)


def test_retrieval_keeps_same_error_code_domains_separate(tokenizer):
    parents, children = childmod.build_children(
        [
            parent(
                "network", "E-12: IP conflict", ["Network", "E-12"], domain="network"
            ),
            parent(
                "display",
                "E-12: waste ink pad",
                ["Display errors"],
                domain="device_display_errors",
            ),
        ],
        tokenizer,
    )
    index = ParentChildIndex(parents, children)
    result = index.resolve(
        [c["child_id"] for c in children], model="L3266", feature_domain="network"
    )
    assert [p["parent_id"] for p in result["parents"]] == ["network"]
    assert result["rejected"][0]["reason"] == "domain_mismatch"
    display = index.resolve(
        [c["child_id"] for c in children],
        model="L3266",
        feature_domain="device_display_errors",
    )
    assert [p["parent_id"] for p in display["parents"]] == ["display"]


def test_retrieval_returns_prerequisites_and_filters_other_platform(tokenizer):
    parents, children = childmod.build_children(
        [
            parent("win", "Check Windows connection first", ["Trouble", "Windows"]),
            parent(
                "win-net",
                "All Windows network steps",
                ["Trouble", "Windows", "Network"],
                contexts=["win"],
            ),
            parent("mac", "Mac instructions", ["Trouble", "Mac OS"]),
        ],
        tokenizer,
    )
    index = ParentChildIndex(parents, children)
    ids = [c["child_id"] for c in children if c["parent_id"] in {"win-net", "mac"}]
    result = index.resolve(ids, model="l3266", platform="Windows")
    assert [p["parent_id"] for p in result["parents"]] == ["win", "win-net"]
    assert result["context_only_parent_ids"] == ["win"]
    assert result["rejected"][0]["reason"] == "platform_mismatch"
    assert result["parents"][1]["content"] == "All Windows network steps"


def test_explicit_model_restriction_and_filename_defaults_survive(tokenizer):
    parents, children = childmod.build_children(
        [
            parent("default", "General instruction"),
            parent("restricted", "ADF specification", models=["L5298"]),
        ],
        tokenizer,
    )
    result = ParentChildIndex(parents, children).resolve(
        [c["child_id"] for c in children], model="L3266"
    )
    assert [p["parent_id"] for p in result["parents"]] == ["default"]
    assert result["rejected"][0]["reason"] == "model_mismatch"


def test_required_context_mismatch_rejects_incomplete_evidence_group(tokenizer):
    parents, children = childmod.build_children(
        [
            parent("restricted", "Prerequisite", ["Guide"], models=["L5298"]),
            parent(
                "general", "Operation", ["Guide", "Operation"], contexts=["restricted"]
            ),
        ],
        tokenizer,
    )
    child = next(c for c in children if c["parent_id"] == "general")
    result = ParentChildIndex(parents, children).resolve(
        [child["child_id"]], model="L3266"
    )
    assert result["parents"] == []
    assert result["rejected"][0]["reason"] == "required_context_model_mismatch"


def test_retrieval_deduplicates_parents_and_preserves_review_flags(tokenizer):
    p = parent(text="Long instructions.\n" * 20)
    p["review_flags"] = ["inline_condition_requires_review"]
    parents, children = childmod.build_children(
        [p], tokenizer, childmod.ChildConfig(50, 5)
    )
    result = ParentChildIndex(parents, children).resolve(
        [c["child_id"] for c in children] + [children[0]["child_id"]], model="L5298"
    )
    assert len(result["parents"]) == 1
    assert result["requires_review"]
    assert len(result["accepted_child_ids"]) == len(children)


@pytest.mark.parametrize("defect", ["missing", "foreign_doc", "cycle"])
def test_context_parent_integrity_is_required(tokenizer, defect):
    first = parent("first", path=["Guide"])
    last = parent("last", path=["Guide", "Operation"], contexts=["first"])
    if defect == "missing":
        last["context_parent_ids"] = ["absent"]
    elif defect == "foreign_doc":
        first["doc_id"] = "other-doc"
    else:
        first["context_parent_ids"] = ["last"]
    parents, children = childmod.build_children([first, last], tokenizer)
    with pytest.raises(ValueError, match="context parent|Context parent"):
        ParentChildIndex(parents, children)


def test_unknown_model_scope_is_not_accepted(tokenizer):
    p = parent()
    p["scope"] = {"status": "unknown", "applicable_models": []}
    parents, children = childmod.build_children([p], tokenizer)
    index = ParentChildIndex(parents, children)
    assert (
        index.resolve([children[0]["child_id"]], model="L5298")["rejected"][0]["reason"]
        == "model_scope_unresolved"
    )
    with pytest.raises(ValueError, match="device model"):
        index.resolve([], model="")


def test_filename_change_changes_child_identity(tokenizer):
    p = parent()
    _, first = childmod.build_children([p], tokenizer)
    p["document_name"] = "L5298.pdf"
    p["scope"]["applicable_models"] = ["L5298"]
    _, second = childmod.build_children([p], tokenizer)
    assert first[0]["child_id"] != second[0]["child_id"]


@pytest.mark.parametrize(
    "title,expected",
    [
        ("BIOS 设置", []),
        ("使用Windows10打印", ["windows"]),
        ("macOS 设置", ["macos"]),
        ("iOS 打印", ["ios"]),
    ],
)
def test_platform_names_do_not_match_inside_unrelated_words(title, expected):
    assert childmod.heading_conditions(parent(path=[title]))["platforms"] == expected
