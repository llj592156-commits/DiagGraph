"""Offline parent candidates retain evidence and refuse unsupported certainty."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lightrag.document_structure import parents as prep

pytestmark = pytest.mark.offline


def block(key, heading, body="", page=1, top=700):
    return {
        "type": "content",
        "blockid": key,
        "heading": heading,
        "content": f"## {heading}" + (f"\n{body}" if body else ""),
        "positions": [
            {"type": "bbox", "anchor": str(page), "range": [50, top, 300, top - 12]}
        ],
    }


def section(path=None, index=0):
    return {
        "section_id": "section-network",
        "heading_path": path or ["网络", "网络连接报告"],
        "start_block_index": index,
        "alignment": "title_and_page",
    }


def meta(count=1):
    return {
        "type": "meta",
        "doc_id": "doc-test",
        "document_name": "L3266_L5298.pdf",
        "blocks": count,
        "bbox_attributes": {"origin": "LEFTBOTTOM"},
    }


def rule(models=None):
    return {
        "id": "adf",
        "path_patterns": ["ADF"],
        "applicable_models": models or ["L5298"],
        "feature": "adf",
        "evidence": {"page": 1, "quote": "Only L5298", "feature_quote": "ADF"},
    }


def test_same_page_repeated_headings_use_coordinates():
    blocks = [block("first", "重复标题", top=700), block("second", "重复标题", top=300)]
    bookmarks = [
        prep.Bookmark("a", None, ("重复标题",), 1, 702),
        prep.Bookmark("b", None, ("重复标题",), 1, 302),
    ]
    sections, issues = prep.align_sections(bookmarks, blocks, {1: 800}, "LEFTBOTTOM")
    assert [row["start_block_id"] for row in sections] == ["first", "second"]
    assert issues == []


def test_title_normalization_handles_width_and_whitespace():
    assert prep.normalize_title("Ｅ－１２： 网络 设置") == prep.normalize_title(
        "E-12 网络设置"
    )


def test_page_number_prevents_matching_table_of_contents_occurrence():
    blocks = [block("toc", "网络", page=1), block("body", "网络", page=3)]
    bookmark = prep.Bookmark("a", None, ("网络",), 3, 700)
    sections, _ = prep.align_sections(
        [bookmark], blocks, {1: 800, 3: 800}, "LEFTBOTTOM"
    )
    assert sections[0]["start_block_id"] == "body"


def test_position_fallback_is_reported_as_uncertain():
    bookmark = prep.Bookmark("a", None, ("未识别标题",), 1, 700)
    sections, issues = prep.align_sections(
        [bookmark], [block("b", "正文")], {1: 800}, "LEFTBOTTOM"
    )
    assert sections[0]["alignment"] == "position_only"
    assert issues[0]["kind"] == "outline_alignment_requires_review"
    parents = prep.build_parents(meta(), [block("b", "正文")], sections, [], 10000)
    assert "position_only_section_alignment" in parents[0]["review_flags"]


def test_missing_positions_do_not_get_a_confident_alignment():
    item = block("b", "网络")
    item["positions"] = []
    sections, issues = prep.align_sections(
        [prep.Bookmark("a", None, ("网络",), 1, 700)], [item], {1: 800}, "LEFTBOTTOM"
    )
    assert sections[0]["alignment"] == "unmatched"
    assert issues


def test_error_code_keeps_solution_and_stops_before_next_topic():
    blocks = [
        block("a", "E-12"),
        block("b", "解决方法：", "检查 IP 地址"),
        block("c", "相关信息", "网络状态页"),
        block("d", "E-13"),
        block("e", "解决方法", "检查路由器"),
        block("f", "其他网络消息", "独立消息"),
    ]
    parents = prep.build_parents(meta(6), blocks, [section()], [], 10000)
    assert [p["source_block_ids"] for p in parents] == [
        ["a", "b", "c"],
        ["d", "e"],
        ["f"],
    ]
    assert parents[0]["fault_codes"] == ["E-12"]
    assert parents[0]["feature_domain"] == "network"
    assert prep.validate_coverage(blocks, parents)["source_content_preserved"] is True


def test_device_error_table_has_a_different_domain_from_network_error():
    item = block(
        "a", "显示屏上显示错误代码", '<table format="json">[["E-12", "废墨"]]</table>'
    )
    parents = prep.build_parents(
        meta(), [item], [section(["故障排除", "显示屏上显示错误代码"])], [], 10000
    )
    assert parents[0]["feature_domain"] == "device_display_errors"
    assert "废墨" in parents[0]["content"]


def test_explicit_scope_does_not_leak_into_next_local_heading():
    blocks = [
        block("a", "仅对于 L5298", "专属操作"),
        block("b", "注意", "限制条件"),
        block("c", "其他操作", "范围尚未明确"),
    ]
    parents = prep.build_parents(
        meta(3), blocks, [section(["部件名称和功能"])], [], 10000
    )
    assert parents[0]["source_block_ids"] == ["a", "b"]
    assert parents[0]["scope"]["applicable_models"] == ["L5298"]
    assert parents[1]["scope"]["status"] == "filename"
    assert parents[1]["scope"]["applicable_models"] == ["L3266", "L5298"]


def test_filename_models_are_default_applicability():
    parents = prep.build_parents(
        meta(), [block("a", "普通操作")], [section()], [], 10000
    )
    assert parents[0]["document_models"] == ["L3266", "L5298"]
    assert parents[0]["scope"]["status"] == "filename"
    assert parents[0]["scope"]["applicable_models"] == ["L3266", "L5298"]


def test_negative_model_qualifier_removes_only_excluded_models():
    blocks = [block("a", "仅对于 L5298 以外的型号", "原文排除 L5298")]
    parents = prep.build_parents(meta(), blocks, [section()], [], 10000)
    assert parents[0]["scope"]["status"] == "explicit"
    assert parents[0]["scope"]["applicable_models"] == ["L3266"]
    assert parents[0]["scope"]["excluded_models"] == ["L5298"]


@pytest.mark.parametrize(
    "first",
    [
        block("a", "仅对于 L5298 以外的型号", "Operation"),
        block("a", "菜单", "本菜单仅适用于 L5298。\nOperation"),
    ],
)
def test_explicit_limits_stop_at_the_next_independent_topic(first):
    blocks = [first, block("b", "其他操作", "Ordinary instructions")]
    parents = prep.build_parents(meta(2), blocks, [section()], [], 10000)
    assert len(parents) == 2
    assert parents[1]["scope"]["applicable_models"] == ["L3266", "L5298"]


def test_warning_marker_stays_with_qualified_operation():
    blocks = [block("a", "仅对于 L5298", "操作"), block("b", "c 重要：", "必要警告")]
    parents = prep.build_parents(meta(2), blocks, [section()], [], 10000)
    assert len(parents) == 1
    assert parents[0]["source_block_ids"] == ["a", "b"]
    assert "必要警告" in parents[0]["content"]


def test_scope_uses_intersection_and_reports_conflicting_evidence():
    blocks = [block("a", "仅对于 L9999", "原文")]
    scope, flags = prep.resolve_scope(["仅对于 L9999"], blocks, [], "L3266_L5298.pdf")
    assert scope["status"] == "conflict"
    assert scope["applicable_models"] == []
    assert len(scope["evidence"]) == 2
    assert flags == ["conflicting_model_scope"]


def test_scope_rule_only_selects_named_chapter_not_incidental_prose():
    item = block("a", "一般维护", "也可清洁 ADF")
    scope, _ = prep.resolve_scope(["一般维护"], [item], [rule()])
    assert scope["status"] == "unknown"


def test_feature_reference_rule_does_not_narrow_filename_default():
    item = block("a", "ADF", "Load originals")
    scope, _ = prep.resolve_scope(["ADF"], [item], [rule()], "L3266_L5298.pdf")
    assert scope["status"] == "filename"
    assert scope["applicable_models"] == ["L3266", "L5298"]
    assert scope["reference_rule_ids"] == ["adf"]


@pytest.mark.parametrize(
    "declaration", ["本菜单仅适用于 L5298。", "仅对于 L5298", "本功能不适用于 L3266。"]
)
def test_opening_declaration_narrows_whole_parent(declaration):
    item = block("a", "Menu", declaration + "\nInstructions")
    parents = prep.build_parents(meta(), [item], [section(["Menu"])], [], 10000)
    assert parents[0]["scope"]["applicable_models"] == ["L5298"]
    assert parents[0]["scope"]["status"] == "explicit"


def test_item_footnote_does_not_narrow_whole_parent():
    item = block("a", "操作面板", "Common controls\n* 仅适用于 L5298")
    parents = prep.build_parents(meta(), [item], [section(["操作面板"])], [], 10000)
    assert parents[0]["scope"]["applicable_models"] == ["L3266", "L5298"]
    assert "inline_condition_requires_review" in parents[0]["review_flags"]


def test_filename_without_models_stays_unknown():
    header = {**meta(), "document_name": "manual.pdf"}
    parents = prep.build_parents(
        header, [block("a", "General")], [section()], [], 10000
    )
    assert parents[0]["scope"]["status"] == "unknown"
    assert parents[0]["scope"]["applicable_models"] == []


def test_repeated_heading_path_restriction_is_inherited():
    item = block("a", "步骤", "Instructions")
    parents = prep.build_parents(
        meta(), [item], [section(["仅对于 L5298", "步骤"])], [], 10000
    )
    assert parents[0]["scope"]["applicable_models"] == ["L5298"]


def test_oversized_table_is_preserved_for_review_instead_of_cut():
    item = block("a", "表格", '<table format="json">[["列", "很长的值"]]</table>')
    parents = prep.build_parents(meta(), [item], [section()], [], 10)
    assert parents[0]["content"] == item["content"]
    assert "oversized_parent_requires_structural_split" in parents[0]["review_flags"]
    assert parents[0]["ready_for_indexing"] is False


def test_chapter_contents_are_separated_from_procedure():
    blocks = [
        block("a", "传真", "操作一...........87\n操作二...........88"),
        block("b", "传真准备", "接好电话线"),
    ]
    parents = prep.build_parents(meta(2), blocks, [section(["传真"])], [], 10000)
    assert [p["content_role"] for p in parents] == ["section_contents", "body"]
    assert parents[1]["content"] == blocks[1]["content"]


def test_duplicate_titles_do_not_overwrite_parent_ids():
    blocks = [block("a", "E-12", "第一处"), block("b", "E-12", "第二处")]
    parents = prep.build_parents(meta(2), blocks, [section()], [], 10000)
    assert len({p["parent_id"] for p in parents}) == 2
    assert parents == prep.build_parents(meta(2), blocks, [section()], [], 10000)


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "reorder", "rewrite"])
def test_coverage_check_rejects_loss_duplication_reordering_and_rewriting(mutation):
    blocks = [block("a", "E-12", "原文一"), block("b", "E-13", "原文二")]
    parents = prep.build_parents(meta(2), blocks, [section()], [], 10000)
    parents = deepcopy(parents)
    if mutation == "drop":
        parents.pop()
    elif mutation == "duplicate":
        parents.append(deepcopy(parents[0]))
    elif mutation == "reorder":
        parents.reverse()
    else:
        parents[0]["content"] = "改写后的正文"
    with pytest.raises(ValueError):
        prep.validate_coverage(blocks, parents)


def write_rules(path, value):
    path.write_text(
        json.dumps(
            {"schema_version": 1, "documents": {"manual.pdf": {"rules": [value]}}}
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "page_text",
    ["Only L5298; ADF", "Only L5298; other feature", "ADF; no model restriction"],
)
def test_rules_validate_both_restriction_and_feature_quotes(tmp_path, page_text):
    reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: page_text)])
    path = tmp_path / "rules.json"
    write_rules(path, rule())
    if page_text == "Only L5298; ADF":
        assert prep.load_rules(path, "manual.pdf", reader)[0]["id"] == "adf"
    else:
        with pytest.raises(ValueError, match="quote not found"):
            prep.load_rules(path, "manual.pdf", reader)


def test_rules_do_not_apply_to_other_documents(tmp_path):
    path = tmp_path / "rules.json"
    write_rules(path, rule())
    assert prep.load_rules(path, "another.pdf", SimpleNamespace(pages=[])) == []


@pytest.mark.parametrize("defect", ["count", "duplicate", "metadata"])
def test_sidecar_validation_refuses_corrupt_input(tmp_path, defect):
    rows = [meta(2), block("a", "标题一"), block("b", "标题二")]
    if defect == "count":
        rows[0]["blocks"] = 3
    elif defect == "duplicate":
        rows[2]["blockid"] = "a"
    else:
        rows[0]["type"] = "content"
    path = tmp_path / "manual.blocks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        prep.read_sidecar(path)


def test_sidecar_content_fingerprint_is_checked(tmp_path):
    header = meta()
    header["document_hash"] = "sha256:wrong"
    path = tmp_path / "manual.blocks.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in [header, block("a", "正文")]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="document_hash"):
        prep.read_sidecar(path)


def test_source_block_crossing_a_bookmark_is_reported_without_losing_content():
    item = block("a", "前一节", page=1, top=700)
    item["positions"].extend(block("b", "后一节正文", page=2, top=500)["positions"])
    parents = prep.build_parents(meta(), [item], [section(["前一节"])], [], 10000)
    bookmark = prep.Bookmark("next", None, ("后一节",), 2, 700)
    issues = prep.report_crossings(
        [bookmark], [item], parents, {1: 800, 2: 800}, "LEFTBOTTOM"
    )
    assert issues[0]["block_id"] == "a"
    assert "source_block_crosses_bookmark" in parents[0]["review_flags"]
    assert parents[0]["content"] == item["content"]


@pytest.fixture
def corpus(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=600, height=800)
    root = writer.add_outline_item("网络", 0)
    writer.add_outline_item("网络连接报告", 0, parent=root)
    pdf = tmp_path / "L3266_L5298.pdf"
    writer.write(pdf)
    parsed = tmp_path / (pdf.name + ".parsed")
    parsed.mkdir()
    blocks = [
        block("a", "网络", top=780),
        block("b", "网络连接报告", top=700),
        block("c", "E-12", top=600),
        block("d", "解决方法", "检查网络地址", top=550),
    ]
    sidecar = parsed / "manual.blocks.jsonl"
    sidecar.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [meta(4), *blocks]),
        encoding="utf-8",
    )
    return pdf, parsed, blocks


def test_complete_preparation_is_deterministic_and_keeps_inputs_immutable(
    corpus, tmp_path
):
    pdf, parsed, blocks = corpus
    original = {p: p.read_bytes() for p in [pdf, *parsed.iterdir()]}
    first, second = tmp_path / "first", tmp_path / "second"
    summary = prep.prepare(pdf, parsed, first)
    prep.prepare(pdf, parsed, second)
    assert summary["source_blocks"] == 4
    assert summary["source_content_preserved"] is True
    assert summary["source_identity_verified"] is False
    for filename in ["parents.jsonl", "sections.jsonl", "issues.jsonl", "summary.json"]:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert all(p.read_bytes() == content for p, content in original.items())
    parents = [
        json.loads(x)
        for x in (first / "parents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert next(p for p in parents if p["fault_codes"] == ["E-12"])[
        "source_block_ids"
    ] == ["c", "d"]
    assert prep.validate_coverage(blocks, parents)["assigned_blocks"] == 4


def test_existing_output_is_never_overwritten(corpus, tmp_path):
    pdf, parsed, _ = corpus
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        prep.prepare(pdf, parsed, output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_shared_organizer_returns_in_memory_result_without_writing(corpus, tmp_path):
    pdf, parsed, _ = corpus
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    result = prep.organize(pdf, parsed)
    assert result.metadata["doc_id"] == "doc-test"
    assert result.summary["parents"] == len(result.parents)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
    summary = prep.write_preparation(result, tmp_path / "result")
    assert summary == result.summary


def test_offline_child_bundle_is_deterministic_and_hash_verified(corpus, tmp_path):
    from lightrag.document_structure.children import prepare_children
    from lightrag.document_structure.retrieval import ParentChildIndex

    pdf, parsed, _ = corpus
    parent_dir = tmp_path / "parents"
    prep.prepare(pdf, parsed, parent_dir)
    tokenizer = SimpleNamespace(
        model_name="characters-test-only", encode=lambda text: list(map(ord, text))
    )
    first, second = tmp_path / "children-one", tmp_path / "children-two"
    summary = prepare_children(parent_dir, first, tokenizer)
    prepare_children(parent_dir, second, tokenizer)
    for filename in ("parents.jsonl", "children.jsonl", "summary.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert summary["children"] > 0
    index = ParentChildIndex.from_directory(first)
    key = next(iter(index.children))
    assert index.resolve([key], model="L5298")["accepted_child_ids"] == [key]
    with pytest.raises(ValueError, match="already exists"):
        prepare_children(parent_dir, first, tokenizer)
    with (first / "children.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="artifact hash"):
        ParentChildIndex.from_directory(first)


def test_child_preparation_rejects_changed_source_sidecar(corpus, tmp_path):
    from lightrag.document_structure.children import prepare_children

    pdf, parsed, _ = corpus
    parent_dir = tmp_path / "parents"
    prep.prepare(pdf, parsed, parent_dir)
    source = next(parsed.glob("*.blocks.jsonl"))
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar hash"):
        prepare_children(parent_dir, tmp_path / "children", SimpleNamespace())
    assert not (tmp_path / "children").exists()


def test_same_filename_with_wrong_pdf_bytes_is_rejected(corpus, tmp_path):
    pdf, parsed, _ = corpus
    raw = tmp_path / (pdf.name + ".docling_raw")
    raw.mkdir()
    manifest = raw / "_manifest.json"
    manifest.write_text(
        json.dumps({"source_content_hash": "sha256:wrong"}), encoding="utf-8"
    )
    output = tmp_path / "parents"
    with pytest.raises(ValueError, match="source hash"):
        prep.prepare(pdf, parsed, output)
    assert not output.exists()


def test_matching_manifest_records_verified_source_identity(corpus, tmp_path):
    pdf, parsed, _ = corpus
    raw = tmp_path / (pdf.name + ".docling_raw")
    raw.mkdir()
    (raw / "_manifest.json").write_text(
        json.dumps({"source_content_hash": prep.file_hash(pdf)}), encoding="utf-8"
    )
    assert (
        prep.prepare(pdf, parsed, tmp_path / "parents")["source_identity_verified"]
        is True
    )


def test_pdf_without_bookmarks_is_not_silently_treated_as_structured(corpus, tmp_path):
    from pypdf import PdfWriter

    pdf, parsed, _ = corpus
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    writer.write(pdf)
    with pytest.raises(ValueError, match="no outline"):
        prep.prepare(pdf, parsed, tmp_path / "parents")


def picture_boundary_fixture():
    item = block(
        "source",
        "Previous operation",
        'Last step.\n<drawing id="cover" format="png" />\nItem........32',
    )
    item["positions"] += block("picture", "", page=2, top=700)["positions"]
    item["positions"] += block("contents", "", page=2, top=600)["positions"]
    raw = {
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "children": [{"$ref": "#/texts/0"}],
                "prov": [
                    {
                        "page_no": 2,
                        "bbox": {
                            "l": 50,
                            "t": 700,
                            "r": 300,
                            "b": 688,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            }
        ],
        "texts": [{"self_ref": "#/texts/0", "text": "Copy", "prov": [{"page_no": 2}]}],
    }
    drawings = {"cover": {"blockid": "source", "self_ref": "#/pictures/0"}}
    bookmark = prep.Bookmark("copy", None, ("Copy",), 2, 700)
    return item, raw, drawings, bookmark


def test_picture_title_recovers_boundary_with_exact_original_offsets():
    item, raw, drawings, bookmark = picture_boundary_fixture()
    fragments, events = prep.split_picture_bookmarks(
        [item], [bookmark], raw, drawings, {1: 800, 2: 800}, "LEFTBOTTOM"
    )
    assert len(fragments) == 2
    assert "".join(f["content"] for f in fragments) == item["content"]
    assert [f["positions"][0]["anchor"] for f in fragments] == ["1", "2"]
    assert events[0]["raw_picture_ref"] == "#/pictures/0"
    sections, issues = prep.align_sections(
        [bookmark], fragments, {1: 800, 2: 800}, "LEFTBOTTOM"
    )
    assert sections[0]["alignment"] == "picture_text_and_page"
    assert issues == []
    parents = prep.build_parents(meta(), fragments, sections, [], 10000)
    assert parents[-1]["content_role"] == "section_contents"
    assert "position_only_section_alignment" not in parents[-1]["review_flags"]
    assert prep.validate_coverage([item], parents)["assigned_fragments"] == 2
    assert [p["source_block_ids"] for p in parents] == [["source"], ["source"]]
    assert (
        prep.report_crossings(
            [bookmark], fragments, parents, {1: 800, 2: 800}, "LEFTBOTTOM"
        )
        == []
    )


@pytest.mark.parametrize("defect", ["text", "page", "bbox", "owner", "ambiguous"])
def test_picture_boundary_requires_consistent_evidence(defect):
    item, raw, drawings, bookmark = picture_boundary_fixture()
    bookmarks = [bookmark]
    if defect == "text":
        raw["texts"][0]["text"] = "Another title"
    elif defect == "page":
        raw["texts"][0]["prov"][0]["page_no"] = 3
    elif defect == "bbox":
        raw["pictures"][0]["prov"][0]["bbox"]["l"] = 51
    elif defect == "owner":
        drawings["cover"]["blockid"] = "another-source"
    else:
        bookmarks.append(prep.Bookmark("duplicate", None, ("Copy",), 2, 700))
    fragments, events = prep.split_picture_bookmarks(
        [item], bookmarks, raw, drawings, {1: 800, 2: 800}, "LEFTBOTTOM"
    )
    assert fragments == [item]
    assert events == []


@pytest.mark.parametrize("defect", ["gap", "overlap", "rewrite", "drop"])
def test_fragment_coverage_rejects_content_damage(defect):
    item, raw, drawings, bookmark = picture_boundary_fixture()
    fragments, _ = prep.split_picture_bookmarks(
        [item], [bookmark], raw, drawings, {1: 800, 2: 800}, "LEFTBOTTOM"
    )
    sections, _ = prep.align_sections(
        [bookmark], fragments, {1: 800, 2: 800}, "LEFTBOTTOM"
    )
    parents = prep.build_parents(meta(), fragments, sections, [], 10000)
    if defect in {"gap", "overlap"}:
        parents[-1]["source_spans"][0]["start"] += 1 if defect == "gap" else -1
    elif defect == "rewrite":
        parents[-1]["content"] += "Changed"
    else:
        parents.pop()
    with pytest.raises(ValueError):
        prep.validate_coverage([item], parents)


def styled_block(key, heading, height, body=""):
    item = block(key, heading, body)
    item["positions"][0]["range"][3] = 700 - height
    return item


def structure_rule():
    return {
        "id": "reviewed",
        "section_path": ["Troubleshooting"],
        "heading_styles": [
            {"bbox_height_pt": 13.99, "depth": 1},
            {"bbox_height_pt": 13.2, "depth": 2},
        ],
    }


def test_structural_split_keeps_platform_context_causes_and_solutions():
    blocks = [
        styled_block("intro", "Troubleshooting", 16),
        styled_block("win", "Windows", 13.99, "Check connection first."),
        styled_block("network-win", "Network unavailable", 13.2),
        styled_block("cause", "Router too far", 11.5),
        styled_block("solution", "解决方案", 10.48, "Move closer."),
        styled_block("note", "注释：", 11.98, "Do not reset during operation."),
        styled_block("usb", "USB unavailable", 13.2, "Reconnect cable."),
        styled_block("mac", "Mac OS", 13.99, "Check connection first."),
        styled_block(
            "network-mac", "Network unavailable", 13.2, "Mac-specific solution."
        ),
    ]
    parents = prep.build_parents(
        meta(9), blocks, [section(["Troubleshooting"])], [], 30, [structure_rule()]
    )
    win = next(
        p for p in parents if p["heading_path"] == ["Troubleshooting", "Windows"]
    )
    network = next(
        p
        for p in parents
        if p["heading_path"] == ["Troubleshooting", "Windows", "Network unavailable"]
    )
    mac = parents[-1]
    assert network["source_block_ids"] == ["network-win", "cause", "solution", "note"]
    assert network["context_parent_ids"] == [win["parent_id"]]
    assert win["parent_id"] not in mac["context_parent_ids"]
    assert parents[0]["source_block_ids"] == ["intro", "win"]
    assert prep.validate_coverage(blocks, parents)["source_content_preserved"] is True


def test_structural_split_preserves_lower_level_poster_steps_and_warnings():
    blocks = [
        styled_block("a", "Poster", 13.99, "Settings"),
        styled_block("b", "c 重要：", 13.99, "Required warning"),
        styled_block("c", "1. Print", 13.99, "First step"),
        styled_block("d", "Align overlap marks", 11.5, "Complete poster procedure"),
        styled_block("e", "Other operation", 13.99, "Next procedure"),
    ]
    parents = prep.build_parents(
        meta(5), blocks, [section(["Troubleshooting"])], [], 10, [structure_rule()]
    )
    assert parents[0]["source_block_ids"] == ["a", "b", "c", "d"]
    assert parents[1]["source_block_ids"] == ["e"]
    assert prep.validate_coverage(blocks, parents)["assigned_blocks"] == 5


def test_structure_rules_only_refine_matching_overlong_sections():
    blocks = [
        styled_block("a", "First", 13.99, "Body"),
        styled_block("b", "Second", 13.99, "Body"),
    ]
    for path, limit in [(["Other chapter"], 1), (["Troubleshooting"], 10000)]:
        parents = prep.build_parents(
            meta(2), blocks, [section(path)], [], limit, [structure_rule()]
        )
        assert len(parents) == 1
        assert parents[0]["structural_evidence"] is None


def test_structure_rules_require_document_allowlist(tmp_path):
    path = tmp_path / "structure.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "document_names": ["manual.pdf"],
                "rules": [structure_rule()],
            }
        ),
        encoding="utf-8",
    )
    assert prep.load_structure_rules(path, "manual.pdf") == [structure_rule()]
    assert prep.load_structure_rules(path, "unreviewed.pdf") == []


def test_heading_coordinate_roundoff_is_not_a_crossing():
    item = block("a", "Same heading", "Body", top=701)
    item["positions"] += block("b", "", top=650)["positions"]
    parents = prep.build_parents(meta(), [item], [section(["Same heading"])], [], 10000)
    bookmark = prep.Bookmark("same", None, ("Same heading",), 1, 700)
    assert (
        prep.report_crossings([bookmark], [item], parents, {1: 800}, "LEFTBOTTOM") == []
    )


def test_empty_source_block_is_still_accounted_for():
    item = block("empty", "")
    item["content"] = ""
    parents = prep.build_parents(meta(), [item], [section()], [], 10000)
    assert prep.validate_coverage([item], parents)["assigned_blocks"] == 1


def test_changed_raw_json_is_rejected_before_using_picture_evidence(corpus, tmp_path):
    pdf, parsed, _ = corpus
    raw = tmp_path / (pdf.name + ".docling_raw")
    raw.mkdir()
    (raw / (pdf.stem + ".json")).write_text("{}", encoding="utf-8")
    (raw / "_manifest.json").write_text(
        json.dumps(
            {
                "source_content_hash": prep.file_hash(pdf),
                "critical_file": {"path": pdf.stem + ".json", "sha256": "sha256:wrong"},
            }
        ),
        encoding="utf-8",
    )
    (parsed / "manual.drawings.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="critical_file"):
        prep.prepare(pdf, parsed, tmp_path / "parents")
    assert not (tmp_path / "parents").exists()
