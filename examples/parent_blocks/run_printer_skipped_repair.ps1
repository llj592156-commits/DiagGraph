param(
    [switch]$Run,
    [switch]$Resume,
    [switch]$PlainJsonRetry,
    [switch]$RepairSavedQuotes,
    [string]$OutputDirectory = 'outputs/printer_skipped_repair/saved_children_v1',
    [string]$PythonExecutable,
    [string]$EnvFile = '.env'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $PythonExecutable) {
    $PythonExecutable = Join-Path $repoRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $PythonExecutable)) { $PythonExecutable = 'python' }
}
$runnerArguments = @(
    '-m', 'lightrag.evaluation.repair_printer_skipped',
    '--env-file', $EnvFile, '--output', $OutputDirectory
)
if ($Run) { $runnerArguments += '--run' }
if ($Resume) { $runnerArguments += '--resume' }
if ($PlainJsonRetry) { $runnerArguments += '--plain-json-retry' }
if ($RepairSavedQuotes) { $runnerArguments += '--repair-saved-quotes' }
Push-Location -LiteralPath $repoRoot
try {
    & $PythonExecutable @runnerArguments
    if ($LASTEXITCODE -eq 2) {
        throw 'Some targeted chunks remain skipped. Inspect summary.json; resume with -Run -Resume after checking the saved errors.'
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'Repair did not complete. Keep the output directory and inspect the traceback; use -Run -Resume after resolving it.'
    }
} finally {
    Pop-Location
}
