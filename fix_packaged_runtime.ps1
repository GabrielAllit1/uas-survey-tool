$ErrorActionPreference = "Stop"

if (-not $env:CONDA_PREFIX) {
    throw "CONDA_PREFIX is not set. Activate uas_survey_tool_v2 first."
}

$src = "$env:CONDA_PREFIX\Library\bin"

$targets = @(
    ".\dist\UAS_Survey_Tool_v2\_internal\PyQt6\Qt6\bin",
    ".\dist\UAS_Survey_Tool_v2_debug\_internal\PyQt6\Qt6\bin"
)

foreach ($target in $targets) {
    if (Test-Path $target) {
        Copy-Item "$src\MSVCP140*.dll" $target -Force
        Copy-Item "$src\VCRUNTIME140*.dll" $target -Force
        Copy-Item "$src\CONCRT140.dll" $target -Force
        Write-Host "Updated VC runtime DLLs in $target"
    }
}
