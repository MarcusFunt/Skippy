Param(
  [string]$Image = "localai/localai:latest-aio-gpu-nvidia-cuda-12",
  [int]$Port = 8080
)

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

Write-Host "Starting LocalAI..."
Write-Host "  Image: $Image"
Write-Host "  Port : $Port"
Write-Host "  Models mount: $Models -> /models"

docker rm -f local-ai 2>$null | Out-Null

docker run -ti --name local-ai `
  -p "$Port`:8080" `
  --gpus all `
  -v "$Models`:/models" `
  $Image
