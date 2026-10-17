param(
    [switch]$RunExperiment,
    [switch]$Resume,
    [string]$OutputDirectory = 'outputs/printer_graph_smoke/manual_run_v2',
    [string]$PythonExecutable,
    [string]$EnvFile = '.env',
    [ValidateRange(256, 16384)][int]$MaxOutputTokens = 4096
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
    '-m', 'lightrag.evaluation.printer_graph_smoke',
    '--bundle-root', 'outputs/printer_parent_child/20261017_v3',
    '--tokenizer-file', 'tmp/parent_blocks_tools/cl100k_base.tiktoken',
    '--env-file', $EnvFile,
    '--output', $OutputDirectory,
    '--max-output-tokens', $MaxOutputTokens
)
if ($Resume) { $runnerArguments += '--resume' }
if ($RunExperiment) { $runnerArguments += '--run' }
Push-Location -LiteralPath $repoRoot
try {
    & $PythonExecutable @runnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Graph smoke experiment exited with code $LASTEXITCODE. Check validation.json and saved raw responses."
    }
} finally {
    Pop-Location
}
