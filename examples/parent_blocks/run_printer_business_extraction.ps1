param(
    [switch]$RunExperiment,
    [switch]$Resume,
    [string]$OutputDirectory = 'outputs/printer_business_extraction/manual_run_v2',
    [string]$PythonExecutable,
    [string]$EnvFile = '.env',
    [ValidateRange(256, 16384)][int]$MaxOutputTokens = 6144
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $PythonExecutable) {
    $projectPython = Join-Path $repoRoot '.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $projectPython) {
        $PythonExecutable = $projectPython
    } else {
        $PythonExecutable = 'python'
    }
}
$runnerArguments = @(
    '-m', 'lightrag.evaluation.printer_business_extraction',
    '--env-file', $EnvFile,
    '--output', $OutputDirectory,
    '--max-output-tokens', $MaxOutputTokens
)
if ($Resume) { $runnerArguments += '--resume' }
if ($RunExperiment) { $runnerArguments += '--run' }
Push-Location -LiteralPath $repoRoot
try {
    $runStartedUtc = [DateTime]::UtcNow
    & $PythonExecutable @runnerArguments
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -eq 2 -and $RunExperiment) {
        $summaryPath = Join-Path $OutputDirectory 'summary.json'
        $validationPath = Join-Path $OutputDirectory 'validation.json'
        $planPath = Join-Path $OutputDirectory 'experiment.json'
        $hasFreshReview = (Test-Path -LiteralPath $summaryPath) -and
            (Test-Path -LiteralPath $validationPath) -and
            (Test-Path -LiteralPath $planPath) -and
            ((Get-Item -LiteralPath $summaryPath).LastWriteTimeUtc -ge $runStartedUtc)
        if ($hasFreshReview) {
            $runSummary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $runPlan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($runSummary.status -eq 'reviewed_with_rejections' -and
                $runSummary.plan_hash -eq $runPlan.plan_hash) {
                Write-Warning "Extraction finished and results were saved. Quality review is required: rejected records=$($runSummary.rejected_records), parse failures=$($runSummary.sample_parse_failures), empty samples=$($runSummary.empty_samples). Read $OutputDirectory/report.md and validation.json before ingestion. No automatic retries were made."
                return
            }
        }
    }
    if ($runExitCode -ne 0) {
        throw "Business extraction failed with exit code $runExitCode. Check the preceding error and any saved raw responses."
    }
} finally {
    Pop-Location
}
