param(
    [switch]$RunExperiment,
    [switch]$Resume,
    [string]$OutputDirectory = 'outputs/printer_retrieval_comparison/manual_run_v2',
    [string]$ReuseEmbeddingDirectory = 'outputs/printer_retrieval_comparison/manual_run_v1',
    [string]$PythonExecutable,
    [string]$TokenizerFile = 'tmp/parent_blocks_tools/cl100k_base.tiktoken',
    [string]$EnvFile = '.env',
    [ValidateRange(1, 64)][int]$BatchSize = 16,
    [ValidateRange(1, 100)][int]$TopK = 5,
    [ValidateRange(1, 100000)][int]$ContextBudget = 8000
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
    '-m', 'lightrag.evaluation.compare_parent_child_retrieval',
    '--legacy-chunks', 'rag_storage/learning_paratera/kv_store_text_chunks.json',
    '--bundle', 'outputs/printer_parent_child/20261017_v3/A',
    '--bundle', 'outputs/printer_parent_child/20261017_v3/B',
    '--bundle', 'outputs/printer_parent_child/20261017_v3/C',
    '--questions', 'examples/parent_blocks/printer_retrieval_cases.jsonl',
    '--tokenizer-file', $TokenizerFile,
    '--env-file', $EnvFile,
    '--output', $OutputDirectory,
    '--batch-size', $BatchSize,
    '--top-k', $TopK,
    '--context-budget', $ContextBudget
)
if ($ReuseEmbeddingDirectory) {
    $runnerArguments += @('--reuse-embedding-directory', $ReuseEmbeddingDirectory)
}
if ($Resume) { $runnerArguments += '--resume' }
if ($RunExperiment) { $runnerArguments += '--allow-remote-embedding' }

Push-Location -LiteralPath $repoRoot
try {
    & $PythonExecutable @runnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Evidence repair comparison exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
