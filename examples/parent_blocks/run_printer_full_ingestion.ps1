param(
    [switch]$Run,
    [switch]$Resume,
    [ValidateRange(1, 16)][int]$ExtractConcurrency = 4,
    [string]$OutputDirectory = 'outputs/printer_full_ingestion/saved_children_v1',
    [string]$SampleRunDirectory = 'outputs/printer_child_ingestion/saved_children_v1',
    [switch]$KeepSampleDocuments,
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
    '--all-children', '--env-file', $EnvFile, '--output', $OutputDirectory,
    '--extract-concurrency', "$ExtractConcurrency"
)
if (-not $KeepSampleDocuments) {
    $runnerArguments += @('--replace-sample-run', $SampleRunDirectory)
}
if ($Run) { $runnerArguments += '--run' }
if ($Resume) { $runnerArguments += '--resume-failed' }
Push-Location -LiteralPath $repoRoot
try {
    & $PythonExecutable @runnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Full ingestion did not complete. Keep the output directory and inspect the error; use -Run -Resume for the same frozen plan after resolving it.'
    }
} finally {
    Pop-Location
}
