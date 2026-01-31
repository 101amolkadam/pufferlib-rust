$ErrorActionPreference = "Stop"

# Read versions from central file
$VersionsFile = Get-Content "VERSIONS" -ErrorAction SilentlyContinue
if (-not $VersionsFile) {
    Write-Host "Error: VERSIONS file not found!" -ForegroundColor Red
    exit 1
}

$LibTorchVersion = ($VersionsFile | Select-String "LIBTORCH_VERSION=(.*)" | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()
$CudaVersion = ($VersionsFile | Select-String "CUDA_VERSION=(.*)" | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()

Write-Host "Configured LibTorch: $LibTorchVersion, CUDA: $CudaVersion"

$LibTorchUrl = "https://download.pytorch.org/libtorch/$CudaVersion/libtorch-win-shared-with-deps-$LibTorchVersion%2B$CudaVersion.zip"
$ZipPath = "libtorch.zip"
$ExtractPath = "$PWD"
$LibTorchPath = "$PWD\libtorch"

if (Test-Path $LibTorchPath) {
    $CurrentVersion = Get-Content "$LibTorchPath\build-version" -ErrorAction SilentlyContinue
    if ($CurrentVersion -notlike "*$LibTorchVersion*") {
        Write-Host "Updating LibTorch from $CurrentVersion to $LibTorchVersion ($CudaVersion)..."
        Remove-Item -Recurse -Force $LibTorchPath
    }
}

if (-not (Test-Path $LibTorchPath)) {
    Write-Host "Downloading LibTorch ($CudaVersion) from $LibTorchUrl..."
    try {
        Invoke-WebRequest -Uri $LibTorchUrl -OutFile $ZipPath
    } catch {
        Write-Host "Stable path failed, trying test path..."
        $LibTorchUrl = "https://download.pytorch.org/libtorch/test/$CudaVersion/libtorch-win-shared-with-deps-$LibTorchVersion%2B$CudaVersion.zip"
        Write-Host "Downloading from: $LibTorchUrl"
        Invoke-WebRequest -Uri $LibTorchUrl -OutFile $ZipPath
    }
    
    Write-Host "Extracting LibTorch..."
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
    
    Remove-Item $ZipPath
} else {
    Write-Host "LibTorch $LibTorchVersion ($CudaVersion) already exists at $LibTorchPath"
}
$env:LIBTORCH = $LibTorchPath
$env:PATH = "$LibTorchPath\lib;$env:PATH"
$env:TORCH_CUDA_VERSION = $CudaVersion

Write-Host "LIBTORCH set to: $env:LIBTORCH"
Write-Host "TORCH_CUDA_VERSION set to: $env:TORCH_CUDA_VERSION"
Write-Host "Verifying cargo check with torch feature..."

cargo check -p pufferlib --features torch
if ($LASTEXITCODE -eq 0) {
    Write-Host "Verification Successful!" -ForegroundColor Green
} else {
    Write-Host "Verification Failed!" -ForegroundColor Red
    exit 1
}
