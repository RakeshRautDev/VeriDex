

$ErrorActionPreference = "Stop"

$HOST = "127.0.0.1"
$PORT = 8000
$URL = "http://$HOST`:$PORT/"

Write-Host "Starting VeriDex Backend Server..." -ForegroundColor Green
Write-Host "Server will run on: $URL" -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-Command", "cd '$PSScriptRoot\VeriDex_WebApp'; python app.py" -WindowStyle Normal -PassThru | Out-Null

Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
$maxAttempts = 60
$attempt = 0
$serverReady = $false

while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri $URL -ErrorAction SilentlyContinue -TimeoutSec 2
        $serverReady = $true
        break
    } catch {
        $attempt++
        Start-Sleep -Seconds 1
    }
}

if ($serverReady) {
    Write-Host "Backend is ready!" -ForegroundColor Green
    Write-Host "Opening frontend in browser..." -ForegroundColor Cyan
    Start-Process $URL
    Write-Host "VeriDex is running!" -ForegroundColor Green
    Write-Host "Press Ctrl+C in the backend window to stop the server." -ForegroundColor Yellow
} else {
    Write-Host "Backend failed to start. Check the server window for errors." -ForegroundColor Red
}
