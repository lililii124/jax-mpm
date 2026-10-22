param([switch]$Render)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Run .\setup.ps1 first." }
& $python -c "import jax; print('Validating JAX', jax.__version__)"
if ($LASTEXITCODE -ne 0) { throw "JAX is not installed; run .\setup.ps1 first." }
$testTemp = Join-Path $PSScriptRoot (".local\pytest-" + [guid]::NewGuid().ToString('N'))
& $python -m pytest -q "--basetemp=$testTemp" --junitxml=outputs/test-results.xml
if ($LASTEXITCODE -ne 0) { throw "Physics tests failed." }
& $python -m examples.gradient
if ($LASTEXITCODE -ne 0) { throw "Gradient example failed." }
& $python -m mpm_jax --scene elastic --frames 2 --substeps 10 --no-render --output outputs/smoke
if ($LASTEXITCODE -ne 0) { throw "2D smoke example failed." }
& $python -m examples.suite --case wave --grid 32 --no-render --output outputs/verify
if ($LASTEXITCODE -ne 0) { throw "3D wave example failed." }
if ($Render) {
    & $python -m mpm_jax.visualization outputs/verify/wave/linear --gif
    if ($LASTEXITCODE -ne 0) { throw "ParaView rendering failed." }
}
