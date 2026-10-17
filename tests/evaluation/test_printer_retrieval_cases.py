"""The frozen printer evaluation set is portable and stays out of the index."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightrag.evaluation.compare_parent_child_retrieval import make_plan

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[2]


def test_printer_fixture_counts_identity_and_provenance():
    path = ROOT / "examples/parent_blocks/printer_retrieval_cases.jsonl"
    manifest = json.loads(
        (path.parent / "printer_retrieval_cases_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    assert len(cases) == len({q["id"] for q in cases}) == manifest["questions"] == 36
    assert (
        sum(q["test_kind"] == "retrieval" for q in cases)
        == manifest["retrieval_questions"]
        == 30
    )
    assert (
        sum(q["test_kind"] == "boundary" for q in cases)
        == manifest["boundary_questions"]
        == 6
    )
    for case in cases:
        assert case["query"].strip()
        if case["test_kind"] == "retrieval":
            assert case["evidence"]
        for evidence in case["evidence"]:
            assert evidence["document"] in {"A", "B", "C"}
            assert evidence["pdf_page"] > 0 and evidence["quote"].strip()


def test_frozen_answers_and_evidence_are_not_embedding_jobs():
    path = ROOT / "examples/parent_blocks/printer_retrieval_cases.jsonl"
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    config = {"base_url": "https://example.test/v1", "model": "fake", "dimension": 2}
    _, jobs = make_plan(
        [{"arm": "child", "text": "Manual input"}],
        cases,
        config,
        {},
        SimpleNamespace(encode=list),
    )
    assert {job["text"] for job in jobs} == {
        "Manual input",
        *(q["query"] for q in cases),
    }
