$ErrorActionPreference = "Stop"

$LibTorchUrl = "https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.10.0%2Bcu126.zip"
$ZipPath = "libtorch.zip"
$ExtractPath = "$PWD"
$LibTorchPath = "$PWD\libtorch"

if (Test-Path $LibTorchPath) {
    $CurrentVersion = Get-Content "$LibTorchPath\build-version" -ErrorAction SilentlyContinue
    if ($CurrentVersion -notlike "*2.10.0*") {
        Write-Host "Updating LibTorch from $CurrentVersion to 2.10.0 (CUDA 12.6)..."
        Remove-Item -Recurse -Force $LibTorchPath
    }
}

if (-not (Test-Path $LibTorchPath)) {
    Write-Host "Downloading LibTorch (CUDA 12.6) from $LibTorchUrl..."
    Invoke-WebRequest -Uri $LibTorchUrl -OutFile $ZipPath
    
    Write-Host "Extracting LibTorch..."
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
    
    Remove-Item $ZipPath
} else {
    Write-Host "LibTorch 2.10.0 (CUDA 12.6) already exists at $LibTorchPath"
}
$env:LIBTORCH = $LibTorchPath
$env:PATH = "$LibTorchPath\lib;$env:PATH"

Write-Host "LIBTORCH set to: $env:LIBTORCH"
Write-Host "Verifying cargo check with torch feature..."

cargo check -p pufferlib --features torch
if ($LASTEXITCODE -eq 0) {
    Write-Host "Verification Successful!" -ForegroundColor Green
} else {
    Write-Host "Verification Failed!" -ForegroundColor Red
    exit 1
}
