$ErrorActionPreference = "Stop"

$LibTorchUrl = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.0%2Bcpu.zip"
$ZipPath = "libtorch.zip"
$ExtractPath = "$PWD"
$LibTorchPath = "$PWD\libtorch"

if (-not (Test-Path $LibTorchPath)) {
    Write-Host "Downloading LibTorch (CPU) from $LibTorchUrl..."
    Invoke-WebRequest -Uri $LibTorchUrl -OutFile $ZipPath
    
    Write-Host "Extracting LibTorch..."
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
    
    Remove-Item $ZipPath
} else {
    Write-Host "LibTorch already exists at $LibTorchPath"
}

$env:LIBTORCH = $LibTorchPath
$env:PATH = "$LibTorchPath\lib;$env:PATH"

Write-Host "LIBTORCH set to: $env:LIBTORCH"
Write-Host "Verifying cargo check with torch feature..."

cargo check --workspace --features torch
if ($LASTEXITCODE -eq 0) {
    Write-Host "Verification Successful!" -ForegroundColor Green
} else {
    Write-Host "Verification Failed!" -ForegroundColor Red
    exit 1
}
