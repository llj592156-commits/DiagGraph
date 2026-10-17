"""The original CLI delegates to the shared organizer."""

import pytest

from lightrag.document_structure import parents
from lightrag.evaluation import prepare_parent_blocks as cli

pytestmark = pytest.mark.offline


def test_cli_uses_shared_preparation():
    assert cli.prepare is parents.prepare


def test_cli_preserves_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli,
        "prepare",
        lambda *args: calls.append(args) or {"parents": 0, "alignment_counts": {}},
    )
    assert (
        cli.main(
            [
                "--pdf",
                str(tmp_path / "manual.pdf"),
                "--parsed-dir",
                str(tmp_path / "parsed"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert calls[0][:3] == (
        tmp_path / "manual.pdf",
        tmp_path / "parsed",
        tmp_path / "out",
    )
