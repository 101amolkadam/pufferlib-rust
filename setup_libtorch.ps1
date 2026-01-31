$ErrorActionPreference = "Stop"

$LibTorchUrl = "https://download.pytorch.org/libtorch/test/cpu/libtorch-win-shared-with-deps-latest.zip"
$ZipPath = "libtorch.zip"
$ExtractPath = "$PWD"
$LibTorchPath = "$PWD\libtorch"

if (Test-Path $LibTorchPath) {
    $CurrentVersion = Get-Content "$LibTorchPath\build-version" -ErrorAction SilentlyContinue
    if ($CurrentVersion -notlike "*2.10.0*") {
        Write-Host "Updating LibTorch from $CurrentVersion to 2.10.0..."
        Remove-Item -Recurse -Force $LibTorchPath
    }
}

if (-not (Test-Path $LibTorchPath)) {
    Write-Host "Downloading LibTorch (CPU 2.10.0) from $LibTorchUrl..."
    Invoke-WebRequest -Uri $LibTorchUrl -OutFile $ZipPath
    
    Write-Host "Extracting LibTorch..."
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
    
    Remove-Item $ZipPath
} else {
    Write-Host "LibTorch 2.10.0 already exists at $LibTorchPath"
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
