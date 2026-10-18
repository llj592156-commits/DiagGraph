param(
    [switch]$Run,
    [switch]$ResumeFailed,
    [string]$OutputDirectory = 'outputs/printer_child_ingestion/saved_children_v1',
    [ValidateRange(1, 8)][int]$PerDocument = 4,
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
    '-m', 'lightrag.evaluation.ingest_printer_children',
    '--env-file', $EnvFile, '--output', $OutputDirectory,
    '--per-document', "$PerDocument"
)
if ($Run) { $runnerArguments += '--run' }
if ($ResumeFailed) { $runnerArguments += '--resume-failed' }
Push-Location -LiteralPath $repoRoot
try {
    & $PythonExecutable @runnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Child ingestion did not complete. Read the preceding error and saved summary/raw responses before retrying.'
    }
} finally {
    Pop-Location
}
