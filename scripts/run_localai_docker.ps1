Param(
  [string]$Image = "",
  [int]$Port = 8080
)

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

# Detect if NVIDIA GPU is available
$GpuFlags = ""
$DefaultImage = "localai/localai:latest-aio-cpu"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $GpuFlags = "--gpus all"
        $DefaultImage = "localai/localai:latest-aio-gpu-nvidia-cuda-12"
    } else {
        Write-Host "NVIDIA GPU detected but nvidia-smi failed. Falling back to CPU."
    }
} else {
    Write-Host "No NVIDIA GPU detected. Falling back to CPU."
}

if (-not $Image) { $Image = $DefaultImage }

Write-Host "Starting LocalAI..."
Write-Host "  Image: $Image"
Write-Host "  Port : $Port"
if ($GpuFlags) {
    Write-Host "  GPU  : $GpuFlags"
} else {
    Write-Host "  GPU  : None"
}
Write-Host "  Models mount: $Models -> /models"

docker rm -f local-ai 2>$null | Out-Null

$DockerArgs = @("run", "-ti", "--name", "local-ai", "-p", "${Port}:8080")
if ($GpuFlags) { $DockerArgs += $GpuFlags.Split(" ") }
$DockerArgs += @("-v", "${Models}:/models", $Image)

docker @DockerArgs
