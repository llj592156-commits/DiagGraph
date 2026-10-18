param(
    [switch]$NoRerank,
    [switch]$DryRun,
    [switch]$RetryErrors,
    [string]$OutputDirectory,
    [string]$PythonExecutable,
    [string]$BaseUrl = 'http://127.0.0.1:9621'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $PythonExecutable) {
    $PythonExecutable = Join-Path $repoRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $PythonExecutable)) { $PythonExecutable = 'python' }
}
if (-not $OutputDirectory) {
    $runName = if ($NoRerank) { 'no_rerank_v2' } else { 'glm_rerank_v1' }
    $OutputDirectory = "outputs/printer_system_retrieval/$runName"
}
$runnerArguments = @(
    'lightrag/evaluation/batch_baseline.py',
    '--dataset', 'examples/parent_blocks/printer_retrieval_cases.jsonl',
    '--output', $OutputDirectory, '--base-url', $BaseUrl,
    '--stage', 'retrieval', '--question-kind', 'retrieval',
    '--modes', 'naive', 'hybrid', 'mix', '--repeats', '1',
    '--top-k', '20', '--chunk-top-k', '5',
    '--max-entity-tokens', '2000', '--max-relation-tokens', '2000',
    '--max-total-tokens', '8000',
    '--label', 'paratera_glm_rerank_candidates20_final5_parent_evidence_v2'
)
if (-not $NoRerank) { $runnerArguments += '--rerank' }
if ($DryRun) { $runnerArguments += '--dry-run' }
if ($RetryErrors) { $runnerArguments += '--retry-errors' }
Push-Location -LiteralPath $repoRoot
try {
    & $PythonExecutable @runnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Retrieval validation has failed or unverified queries. Inspect summary.json, rerank_applied metadata and server logs before retrying.'
    }
} finally {
    Pop-Location
}
