# Kuchiko - Windows Setup Script
# PowerShell script for Windows environments

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Color {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

Write-Host ""
Write-Color "========================================" "Blue"
Write-Color "   Kuchiko Setup Script (Windows)" "Blue"
Write-Color "========================================" "Blue"
Write-Host ""

# Step 1: Check if .env file exists
Write-Color "[1/5] Checking environment file..." "Yellow"
if (-not (Test-Path ".env")) {
    Write-Color "Error: .env file not found!" "Red"
    Write-Host ""
    Write-Host "Please create a .env file first:"
    Write-Host "  1. Create the file: notepad .env"
    Write-Host "  2. Add your API keys (see README for template)"
    Write-Host ""
    Write-Host "Required values:"
    Write-Host "  NVIDIA_API_KEY=your_nvidia_api_key_here"
    Write-Host "  TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here"
    Write-Host "  PDF_FILE=kg.pdf"
    Write-Host ""
    exit 1
}
Write-Color "OK .env file found" "Green"

# Step 2: Validate environment variables
Write-Color "[2/5] Validating environment variables..." "Yellow"
$envContent = Get-Content ".env" | Where-Object { $_ -match "=" }
$envVars = @{}
foreach ($line in $envContent) {
    if ($line -match "^([^=]+)=(.*)$") {
        $envVars[$matches[1].Trim()] = $matches[2].Trim()
    }
}

if (-not $envVars["NVIDIA_API_KEY"] -or $envVars["NVIDIA_API_KEY"] -eq "your_nvidia_api_key_here") {
    Write-Color "Error: NVIDIA_API_KEY is not set in .env file" "Red"
    exit 1
}

if (-not $envVars["TELEGRAM_BOT_TOKEN"] -or $envVars["TELEGRAM_BOT_TOKEN"] -eq "your_telegram_bot_token_here") {
    Write-Color "Error: TELEGRAM_BOT_TOKEN is not set in .env file" "Red"
    exit 1
}

$pdfFile = if ($envVars["PDF_FILE"]) { $envVars["PDF_FILE"] } else { "kg.pdf" }
if (-not (Test-Path $pdfFile)) {
    Write-Color "Error: PDF file '$pdfFile' not found!" "Red"
    Write-Host "Please place your knowledge base PDF in the project folder."
    exit 1
}
Write-Color "OK Environment variables validated" "Green"
Write-Color "OK PDF file found: $pdfFile" "Green"

# Step 3: Check and install Docker if needed
Write-Color "[3/5] Checking Docker installation..." "Yellow"

$dockerInstalled = $false
try {
    $null = docker --version 2>$null
    $dockerInstalled = $true
} catch {
    $dockerInstalled = $false
}

if (-not $dockerInstalled) {
    Write-Color "Docker is not installed. Attempting to install..." "Yellow"

    # Check for winget
    $wingetAvailable = $false
    try {
        $null = winget --version 2>$null
        $wingetAvailable = $true
    } catch {
        $wingetAvailable = $false
    }

    # Check for chocolatey
    $chocoAvailable = $false
    try {
        $null = choco --version 2>$null
        $chocoAvailable = $true
    } catch {
        $chocoAvailable = $false
    }

    if ($wingetAvailable) {
        Write-Color "Installing Docker Desktop via winget..." "Yellow"
        winget install -e --id Docker.DockerDesktop --accept-source-agreements --accept-package-agreements
        Write-Color "OK Docker Desktop installed" "Green"
        Write-Color "Please start Docker Desktop from the Start Menu, then run this script again." "Yellow"
        exit 0
    } elseif ($chocoAvailable) {
        Write-Color "Installing Docker Desktop via Chocolatey..." "Yellow"
        choco install docker-desktop -y
        Write-Color "OK Docker Desktop installed" "Green"
        Write-Color "Please start Docker Desktop from the Start Menu, then run this script again." "Yellow"
        exit 0
    } else {
        Write-Color "Error: Cannot auto-install Docker. Please install manually:" "Red"
        Write-Host ""
        Write-Host "Option 1: Download from https://www.docker.com/products/docker-desktop/"
        Write-Host "Option 2: Install winget, then run: winget install Docker.DockerDesktop"
        Write-Host "Option 3: Install Chocolatey, then run: choco install docker-desktop"
        Write-Host ""
        exit 1
    }
}
Write-Color "OK Docker is installed" "Green"

# Step 4: Check if Docker daemon is running
Write-Color "[4/5] Checking Docker daemon..." "Yellow"

$dockerRunning = $false
try {
    $null = docker info 2>$null
    $dockerRunning = $true
} catch {
    $dockerRunning = $false
}

if (-not $dockerRunning) {
    Write-Color "Docker daemon is not running." "Yellow"
    Write-Host ""
    Write-Host "Please start Docker Desktop:"
    Write-Host "  1. Open Docker Desktop from the Start Menu"
    Write-Host "  2. Wait for it to fully start (icon in system tray)"
    Write-Host "  3. Run this script again"
    Write-Host ""

    # Try to start Docker Desktop
    $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerPath) {
        Write-Color "Attempting to start Docker Desktop..." "Yellow"
        Start-Process $dockerPath
        Write-Host "Docker Desktop is starting. Please wait for it to fully load, then run this script again."
    }
    exit 1
}
Write-Color "OK Docker daemon is running" "Green"

# Step 5: Start the services
Write-Color "[5/5] Starting Kuchiko services..." "Yellow"
Write-Host ""

# Determine docker-compose command
$composeCmd = "docker-compose"
try {
    $null = docker compose version 2>$null
    $composeCmd = "docker compose"
} catch {
    $composeCmd = "docker-compose"
}

# Stop any existing containers
Write-Host "Stopping any existing containers..."
& cmd /c "$composeCmd down 2>nul" 2>$null

# Build and start
Write-Host ""
Write-Color "Building and starting containers..." "Blue"
Write-Color "(This may take a few minutes on first run)" "Yellow"
Write-Host ""

& cmd /c "$composeCmd up -d --build"

Write-Host ""
Write-Color "========================================" "Green"
Write-Color "   Kuchiko is starting up!" "Green"
Write-Color "========================================" "Green"
Write-Host ""
Write-Host "The bot is initializing. This includes:"
Write-Host "  1. Starting Memgraph database"
Write-Host "  2. Processing your PDF into a knowledge graph"
Write-Host "  3. Building the search index"
Write-Host "  4. Starting the Telegram bot"
Write-Host ""
Write-Color "First-time setup can take 5-10 minutes." "Yellow"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  View logs:     $composeCmd logs -f"
Write-Host "  Stop bot:      $composeCmd down"
Write-Host "  Restart:       $composeCmd restart"
Write-Host ""
Write-Color "Watching logs (Ctrl+C to exit)..." "Blue"
Write-Host ""

# Follow logs
& cmd /c "$composeCmd logs -f"
