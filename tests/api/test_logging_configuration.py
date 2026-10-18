"""Exercise startup logging against real, initially absent directories."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.offline


@pytest.mark.parametrize("location", ["relative", "absolute", "default"])
def test_startup_logging_creates_its_directory(tmp_path, location):
    """Configure and write a log without requiring a pre-existing LOG_DIR."""
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo_root), env.get("PYTHONPATH")])
    )
    env.pop("LOG_DIR", None)
    if location == "relative":
        env["LOG_DIR"] = "./log"
        log_dir = tmp_path / "log"
    elif location == "absolute":
        log_dir = tmp_path / "service" / "nested" / "log"
        env["LOG_DIR"] = str(log_dir)
    else:
        log_dir = tmp_path

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging; "
            "from lightrag.api.lightrag_server import configure_logging; "
            "configure_logging(); "
            "logging.getLogger('lightrag').info('startup logging regression'); "
            "logging.shutdown()",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "startup logging regression" in (log_dir / "lightrag.log").read_text(
        encoding="utf-8"
    )
