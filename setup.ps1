# setup.ps1 - Windows Quick Start Script
#
# This script fulfills the requirement:
# "Provide a venv environment for running your code, and ensure your service
# should start cleanly with a single uvicorn command"
#
# It creates a virtual environment, installs dependencies, and starts the server.

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Semantic Search Venv Setup & Start" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$venvPath = ".\.venv"

# 1. Check if Python is installed
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# 2. Create the virtual environment if it doesn't exist
if (-not (Test-Path $venvPath)) {
    Write-Host "`n[1/3] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv $venvPath
    Write-Host "✅ Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "`n[1/3] Virtual environment already exists. Skipping creation." -ForegroundColor Green
}

# 3. Activate venv and install requirements
Write-Host "`n[2/3] Installing/Updating dependencies from requirements.txt..." -ForegroundColor Yellow
# We run python inside the venv directly rather than dealing with Activate.ps1 execution policies
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$venvPath\Scripts\python.exe" -m pip install -r requirements.txt
Write-Host "✅ Dependencies installed." -ForegroundColor Green

# 4. Start the server
Write-Host "`n[3/3] Starting the FastAPI server..." -ForegroundColor Yellow
Write-Host "The server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation (Swagger):     http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

& "$venvPath\Scripts\uvicorn.exe" app.main:app --host 127.0.0.1 --port 8000
