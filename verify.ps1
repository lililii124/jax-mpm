$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Run .\setup.ps1 first." }
& $python -c "import jax; print('Validating JAX', jax.__version__)"
if ($LASTEXITCODE -ne 0) { throw "JAX is not installed; run .\setup.ps1 first." }
& $python -m pytest -q --junitxml=outputs/test-results.xml
if ($LASTEXITCODE -ne 0) { throw "Physics tests failed." }
& $python -m examples.gradient
if ($LASTEXITCODE -ne 0) { throw "Gradient example failed." }
& $python -m mpm_jax --scene elastic --gif
if ($LASTEXITCODE -ne 0) { throw "Elastic example failed." }
& $python -m mpm_jax --scene dam-break --gif
if ($LASTEXITCODE -ne 0) { throw "Dam-break example failed." }
