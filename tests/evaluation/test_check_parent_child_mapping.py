"""Mapping reports fail visibly and cannot overwrite an earlier experiment."""

import json

import pytest

from lightrag.document_structure.retrieval import ParentChildIndex
from lightrag.evaluation import check_parent_child_mapping as check

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("passed,expected_exit", [(True, 0), (False, 1)])
def test_mapping_cli_reports_result_and_refuses_overwrite(
    monkeypatch, tmp_path, passed, expected_exit
):
    monkeypatch.setattr(
        check.ParentChildIndex, "from_directory", lambda directory: object()
    )
    monkeypatch.setattr(
        check,
        "check_printer_mappings",
        lambda index: [{"case": "example", "passed": passed}],
    )
    output = tmp_path / "report.json"
    args = ["--bundle-dir", str(tmp_path / "bundle"), "--output", str(output)]
    assert check.main(args) == expected_exit
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mapping_only"] is True
    assert report["passed"] == int(passed)
    with pytest.raises(ValueError, match="already exists"):
        check.main(args)


def test_mapping_checks_require_actual_examples():
    with pytest.raises(ValueError, match="Expected one printer example parent"):
        check.check_printer_mappings(ParentChildIndex([], []))
