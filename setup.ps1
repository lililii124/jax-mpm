param(
    [string]$PythonPath = "python",
    [string]$JaxSource = "",
    [string]$WheelDirectory = ""
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$envDirectory = Join-Path $PSScriptRoot ".venv312"
$envPython = Join-Path $envDirectory "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $envPython)) {
    & $PythonPath -c "import sys; assert sys.version_info >= (3, 12), 'Python >= 3.12 required'"
    if ($LASTEXITCODE -ne 0) { throw "Unsupported Python interpreter." }
    & $PythonPath -m venv $envDirectory
    if ($LASTEXITCODE -ne 0) { throw "Could not create the project environment." }
}

$pipArguments = @("--disable-pip-version-check", "--no-input")
if ($WheelDirectory) {
    $pipArguments += @("--no-index", "--find-links", (Resolve-Path -LiteralPath $WheelDirectory).Path)
} else {
    $pipArguments += @("--index-url", "https://pypi.org/simple")
}
if ($JaxSource) {
    & $envPython -m pip install @pipArguments (Resolve-Path -LiteralPath $JaxSource).Path
    if ($LASTEXITCODE -ne 0) { throw "JAX source installation failed." }
}
& $envPython -m pip install @pipArguments -e ".[test]"
if ($LASTEXITCODE -ne 0) { throw "Installation failed." }
& $envPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }
& $envPython -c "import jax; print('JAX', jax.__version__); print(jax.devices())"
if ($LASTEXITCODE -ne 0) { throw "JAX import failed." }
Write-Host "Installed in .venv312. Run .\verify.ps1 to test."
