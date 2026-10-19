"""Offline integrity reports detect damaged attribution and evidence bodies."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lightrag.document_structure.children import ChildConfig, build_children
from lightrag.evaluation import audit_parent_child_integrity as audit

pytestmark = pytest.mark.offline


@pytest.fixture
def corpus(tmp_path):
    tokenizer = SimpleNamespace(model_name="test", encode=lambda s: list(map(ord, s)))
    positions = [{"anchor": "2", "type": "bbox", "range": [0, 10, 20, 0]}]
    parent = {
        "parent_id": "p",
        "doc_id": "d",
        "document_name": "manual.pdf",
        "content": "Check the nozzles. Do not interrupt cleaning.",
        "content_role": "body",
        "heading_path": ["Maintenance"],
        "scope": {"status": "filename", "applicable_models": ["L4263"]},
        "feature_domain": "unknown",
        "context_parent_ids": [],
        "review_flags": [],
        "source_block_ids": ["b"],
        "source_fragment_ids": ["f"],
        "source_spans": [{"block_id": "b", "fragment_id": "f", "start": 0, "end": 43}],
        "source_positions": [
            {"block_id": "b", "fragment_id": "f", "positions": positions}
        ],
        "pages": [2],
    }
    parent["source_spans"][0]["end"] = len(parent["content"])
    config = ChildConfig(500, 60)
    parents, children = build_children([parent], tokenizer, config)
    parent, child = parents[0], children[0]
    key = "d-pc-" + child["child_id"]
    stored_parent = {
        **deepcopy(parent),
        "source_doc_id": "d",
        "indexed_chunk_ids": [key],
    }
    meta = {
        k: deepcopy(v)
        for k, v in child.items()
        if k not in {"content", "retrieval_text"}
    }
    meta["source_doc_id"] = "d"
    docs = {
        "d": {
            "file_path": "manual.pdf",
            "parent_child": {"doc_id": "d", "parents": {"p": stored_parent}},
        }
    }
    rows = {
        key: {
            "chunk_id": "pc-" + child["child_id"],
            "full_doc_id": "d",
            "file_path": "manual.pdf",
            "content": child["content"],
            "parent_child": meta,
        }
    }
    mapping = {
        "source_chunk_id": key,
        "doc_id": "d",
        "parent_id": "p",
        "pages": [2],
        "evidence_level": "full_parent",
        "source_chunk_ids": [key],
    }
    chunk = {"chunk_id": key, "content": parent["content"], "file_path": "manual.pdf"}
    response = {
        "status": "success",
        "data": {"chunks": [chunk]},
        "metadata": {"parent_child": {"parents": [mapping]}},
    }

    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"type": "meta", "blocks": 1})
        + "\n"
        + json.dumps(
            {
                "type": "content",
                "blockid": "b",
                "content": parent["content"],
                "positions": positions,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"test-only-placeholder")
    original = tmp_path / "original"
    original.mkdir()
    (original / "parents.jsonl").write_text(json.dumps(parent) + "\n", encoding="utf-8")
    write_json(
        original / "summary.json",
        {
            "pdf_pages": 2,
            "inputs": {
                "pdf": str(pdf),
                "pdf_sha256": audit.digest(pdf),
                "sidecar": str(source),
                "sidecar_sha256": audit.digest(source),
            },
        },
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name, value in (("parents.jsonl", parent), ("children.jsonl", child)):
        (bundle / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
    write_json(
        bundle / "summary.json",
        {
            "schema_version": "parent-child-draft-v3",
            "doc_id": "d",
            "document_name": "manual.pdf",
            "parents": 1,
            "children": 1,
            "config": {"target_tokens": 500, "overlap_tokens": 60},
            "inputs": {
                "parent_dir": str(original),
                "parents_sha256": audit.digest(original / "parents.jsonl"),
                "parent_summary_sha256": audit.digest(original / "summary.json"),
            },
            "outputs": {
                n: audit.digest(bundle / n) for n in ("parents.jsonl", "children.jsonl")
            },
        },
    )
    write_json(tmp_path / "docs.json", docs)
    write_json(tmp_path / "rows.json", rows)
    write_json(tmp_path / "results.jsonl", {"response": response})
    args = SimpleNamespace(
        bundle=[bundle],
        full_docs=tmp_path / "docs.json",
        text_chunks=tmp_path / "rows.json",
        retrieval_results=tmp_path / "results.jsonl",
    )
    return SimpleNamespace(
        parent=parent,
        child=child,
        docs=docs,
        rows=rows,
        key=key,
        mapping=mapping,
        chunk=chunk,
        args=args,
        tokenizer=tokenizer,
    )


def test_end_to_end_read_only_report(corpus, tmp_path):
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = audit.run_audit(corpus.args, corpus.tokenizer)
    assert report["passed"], report["checks"]
    assert report["counts"]["retrieved_evidence"]["passed"] == 1
    assert report["corpus"]["children"] == 1
    assert before == {p: p.read_bytes() for p in before}
    output = tmp_path / "report"
    audit.write_report(output, report)
    assert json.loads((output / "report.json").read_text())["passed"]
    with pytest.raises(FileExistsError):
        audit.write_report(output, report)


@pytest.mark.parametrize(
    "damage", ["pages", "span", "source_position", "orphan", "lost_text", "parent_text"]
)
def test_bundle_corruption_fails_even_with_updated_artifact_hash(corpus, damage):
    bundle = corpus.args.bundle[0]
    parent, child = deepcopy(corpus.parent), deepcopy(corpus.child)
    if damage == "pages":
        child["pages"] = [1]
    elif damage == "span":
        parent["source_spans"][0]["start"] = 1
    elif damage == "source_position":
        parent["source_positions"][0]["positions"][0]["anchor"] = "1"
    elif damage == "orphan":
        child["parent_id"] = "missing"
    elif damage == "lost_text":
        child["content"] = child["content"][:-1]
    else:
        parent["content"] += " changed"
    for name, value in (("parents.jsonl", parent), ("children.jsonl", child)):
        (bundle / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
    summary = json.loads((bundle / "summary.json").read_text())
    summary["outputs"] = {n: audit.digest(bundle / n) for n in summary["outputs"]}
    (bundle / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert not audit.run_audit(corpus.args, corpus.tokenizer)["passed"]


@pytest.mark.parametrize("damage", ["text", "filename", "owner", "links", "pages"])
def test_stored_corruption_fails(corpus, damage):
    if damage == "links":
        corpus.docs["d"]["parent_child"]["parents"]["p"]["indexed_chunk_ids"] = []
    else:
        row = corpus.rows[corpus.key]
        if damage == "text":
            row["content"] = "truncated"
        elif damage == "filename":
            row["file_path"] = "other.pdf"
        elif damage == "owner":
            row["full_doc_id"] = "other"
        else:
            row["parent_child"]["pages"] = [1]
    corpus.args.full_docs.write_text(json.dumps(corpus.docs), encoding="utf-8")
    corpus.args.text_chunks.write_text(json.dumps(corpus.rows), encoding="utf-8")
    assert not audit.run_audit(corpus.args, corpus.tokenizer)["passed"]


@pytest.mark.parametrize(
    "damage", ["body", "pages", "support", "filename", "level", "prerequisite"]
)
def test_saved_evidence_corruption_fails(corpus, damage):
    if damage == "body":
        corpus.chunk["content"] = corpus.chunk["content"][:-1]
    elif damage == "pages":
        corpus.mapping["pages"] = [1]
    elif damage == "support":
        corpus.mapping["source_chunk_ids"] = ["missing"]
    elif damage == "filename":
        corpus.chunk["file_path"] = "other.pdf"
    elif damage == "level":
        corpus.mapping["evidence_level"] = "chunk"
    else:
        corpus.rows[corpus.key]["parent_child"]["context_parent_ids"] = ["missing"]
    with pytest.raises((ValueError, KeyError)):
        audit.check_evidence(corpus.chunk, [corpus.mapping], corpus.docs, corpus.rows)


def test_child_fallback_preserves_prerequisite_and_notes(corpus):
    ancestor = {
        **deepcopy(corpus.parent),
        "parent_id": "ancestor",
        "content": "Safety prerequisite",
    }
    corpus.docs["d"]["parent_child"]["parents"]["ancestor"] = ancestor
    corpus.rows[corpus.key]["parent_child"]["context_parent_ids"] = ["ancestor"]
    corpus.rows[corpus.key]["content"] += "\n\nRequired table note"
    corpus.mapping["evidence_level"] = "child_with_prerequisites"
    mappings = [corpus.mapping, {**corpus.mapping, "parent_id": "ancestor"}]
    corpus.chunk["content"] = (
        corpus.rows[corpus.key]["content"] + "\n\nSafety prerequisite"
    )
    audit.check_evidence(corpus.chunk, mappings, corpus.docs, corpus.rows)
    corpus.chunk["content"] = corpus.parent["content"] + "\n\nSafety prerequisite"
    with pytest.raises(ValueError, match="truncated or altered"):
        audit.check_evidence(corpus.chunk, mappings, corpus.docs, corpus.rows)


def test_hash_and_concurrent_change_are_detected(tmp_path):
    path = tmp_path / "input"
    path.write_text("original")
    result = audit.Audit()
    with pytest.raises(ValueError, match="hash mismatch"):
        result.track(path, "wrong")
    result.track(path)
    path.write_text("changed")
    assert not result.finish()["passed"]


@pytest.mark.parametrize("passed,expected", [(True, 0), (False, 1)])
def test_cli_exit_status_and_no_overwrite(monkeypatch, tmp_path, passed, expected):
    report = {"passed": passed, "counts": {}, "limitations": [], "checks": []}
    monkeypatch.setattr(audit, "run_audit", lambda args: report)
    args = [
        "--bundle",
        "unused",
        "--tokenizer",
        "unused",
        "--full-docs",
        "unused",
        "--text-chunks",
        "unused",
        "--retrieval-results",
        "unused",
        "--output",
        str(tmp_path / "out"),
    ]
    assert audit.main(args) == expected
    with pytest.raises(SystemExit) as exc:
        audit.main(args)
    assert exc.value.code == 2


def test_cli_missing_inputs_produce_failure_report(tmp_path):
    output = tmp_path / "out"
    args = [
        "--bundle",
        "unused",
        "--tokenizer",
        str(tmp_path / "missing"),
        "--full-docs",
        "unused",
        "--text-chunks",
        "unused",
        "--retrieval-results",
        "unused",
        "--output",
        str(output),
    ]
    assert audit.main(args) == 1
    report = json.loads((output / "report.json").read_text())
    assert not report["passed"]
    assert "FileNotFoundError" in report["checks"][0]["error"]
