param(
    [switch]$RunExperiment,
    [switch]$Resume,
    [string]$OutputDirectory = 'outputs/printer_business_extraction/manual_run_v1',
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
    & $PythonExecutable @runnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Business extraction exited with code $LASTEXITCODE. Check validation.json and saved raw responses."
    }
} finally {
    Pop-Location
}
