# Kuchiko - Windows Stop Script
# PowerShell script to stop Kuchiko services

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Stopping Kuchiko services..." -ForegroundColor Yellow

# Determine docker-compose command
$composeCmd = "docker-compose"
try {
    $null = docker compose version 2>$null
    $composeCmd = "docker compose"
} catch {
    $composeCmd = "docker-compose"
}

# Stop containers
& cmd /c "$composeCmd down"

Write-Host ""
Write-Host "Kuchiko services stopped." -ForegroundColor Green
Write-Host ""
