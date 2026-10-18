"""Exercise PowerShell quality warnings without running Python or any provider."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("mode", ["review", "crash", "stale", "mismatch"])
def test_wrapper_distinguishes_fresh_quality_review_from_execution_failure(
    tmp_path, mode
):
    shell = shutil.which("powershell.exe")
    if os.name != "nt" or shell is None:
        pytest.skip("The Windows PowerShell wrapper requires Windows PowerShell")
    output = tmp_path / "output"
    output.mkdir()
    helper = tmp_path / "fixture.ps1"
    helper.write_text(
        """$ErrorActionPreference = 'Stop'
if ($env:PRINTER_WRAPPER_TEST_MODE -eq 'crash') { exit 1 }
$folder = $env:PRINTER_WRAPPER_TEST_OUTPUT
$plan = @{ plan_hash = 'same' }
$summary = @{
    status = 'reviewed_with_rejections'; plan_hash = 'same';
    rejected_records = 9; sample_parse_failures = 0; empty_samples = 0
}
if ($env:PRINTER_WRAPPER_TEST_MODE -eq 'mismatch') { $summary.plan_hash = 'other' }
$plan | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'experiment.json') -Encoding UTF8
'{}' | Set-Content -LiteralPath (Join-Path $folder 'validation.json') -Encoding UTF8
$summaryPath = Join-Path $folder 'summary.json'
$summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding UTF8
if ($env:PRINTER_WRAPPER_TEST_MODE -eq 'stale') {
    (Get-Item -LiteralPath $summaryPath).LastWriteTimeUtc = [DateTime]::UtcNow.AddHours(-1)
}
exit 2
""",
        encoding="utf-8",
    )
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        '@echo off\r\npowershell.exe -NoProfile -NonInteractive -File "%~dp0fixture.ps1"\r\n',
        encoding="utf-8",
    )
    wrapper = (
        Path(__file__).resolve().parents[2]
        / "examples/parent_blocks/run_printer_business_extraction.ps1"
    )

    def literal(path):
        return "'" + str(path).replace("'", "''") + "'"

    command = (
        f"& {literal(wrapper)} -RunExperiment -PythonExecutable {literal(fake_python)} "
        f"-OutputDirectory {literal(output)}; if ($?) {{ exit 0 }} else {{ exit 1 }}"
    )
    environment = {
        **os.environ,
        "PRINTER_WRAPPER_TEST_MODE": mode,
        "PRINTER_WRAPPER_TEST_OUTPUT": str(output),
    }
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    text = result.stdout + result.stderr
    if mode == "review":
        assert result.returncode == 0, text
        assert "Quality review is required" in text
        assert "rejected records=9" in text
    else:
        assert result.returncode != 0, text
        assert "Business extraction failed" in text
        assert "Quality review is required" not in text
