#!/bin/bash

# Kuchiko - One-Click Setup Script
# This script handles everything after you've created your .env file

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   🌿 Kuchiko Setup Script 🌿${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Check if .env file exists
echo -e "${YELLOW}[1/5] Checking environment file...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo ""
    echo "Please create a .env file first:"
    echo "  1. Copy the example file:  cp .env.example .env"
    echo "  2. Edit it with your API keys:  nano .env"
    echo ""
    echo "Required values:"
    echo "  - NVIDIA_API_KEY (get from https://build.nvidia.com/)"
    echo "  - TELEGRAM_BOT_TOKEN (get from @BotFather on Telegram)"
    echo ""
    exit 1
fi
echo -e "${GREEN}✓ .env file found${NC}"

# Step 2: Validate required environment variables
echo -e "${YELLOW}[2/5] Validating environment variables...${NC}"
source .env

if [ -z "$NVIDIA_API_KEY" ] || [ "$NVIDIA_API_KEY" = "your_nvidia_api_key_here" ]; then
    echo -e "${RED}❌ Error: NVIDIA_API_KEY is not set in .env file${NC}"
    exit 1
fi

if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ "$TELEGRAM_BOT_TOKEN" = "your_telegram_bot_token_here" ]; then
    echo -e "${RED}❌ Error: TELEGRAM_BOT_TOKEN is not set in .env file${NC}"
    exit 1
fi

# Check PDF file
PDF_FILE="${PDF_FILE:-kg.pdf}"
if [ ! -f "$PDF_FILE" ]; then
    echo -e "${RED}❌ Error: PDF file '$PDF_FILE' not found!${NC}"
    echo "Please place your knowledge base PDF in the project folder."
    exit 1
fi
echo -e "${GREEN}✓ Environment variables validated${NC}"
echo -e "${GREEN}✓ PDF file found: $PDF_FILE${NC}"

# Step 3: Check if Docker is installed
echo -e "${YELLOW}[3/5] Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker is not installed!${NC}"
    echo ""
    echo "Please install Docker first:"
    echo "  - macOS: brew install --cask docker"
    echo "  - Windows: Download from https://docker.com"
    echo "  - Linux: sudo apt-get install docker.io docker-compose"
    echo ""
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Error: Docker Compose is not installed!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

# Step 4: Check if Docker daemon is running
echo -e "${YELLOW}[4/5] Checking Docker daemon...${NC}"
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Error: Docker daemon is not running!${NC}"
    echo ""
    echo "Please start Docker:"
    echo "  - macOS/Windows: Open Docker Desktop"
    echo "  - Linux: sudo systemctl start docker"
    echo ""
    exit 1
fi
echo -e "${GREEN}✓ Docker daemon is running${NC}"

# Step 5: Start the services
echo -e "${YELLOW}[5/5] Starting Kuchiko services...${NC}"
echo ""

# Determine docker-compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Stop any existing containers
echo "Stopping any existing containers..."
$COMPOSE_CMD down 2>/dev/null || true

# Build and start
echo ""
echo -e "${BLUE}Building and starting containers...${NC}"
echo -e "${YELLOW}(This may take a few minutes on first run)${NC}"
echo ""

$COMPOSE_CMD up -d --build

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   ✅ Kuchiko is starting up!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "The bot is initializing. This includes:"
echo "  1. Starting Memgraph database"
echo "  2. Processing your PDF into a knowledge graph"
echo "  3. Building the search index"
echo "  4. Starting the Telegram bot"
echo ""
echo -e "${YELLOW}First-time setup can take 5-10 minutes.${NC}"
echo ""
echo "Useful commands:"
echo "  View logs:     $COMPOSE_CMD logs -f"
echo "  Stop bot:      $COMPOSE_CMD down"
echo "  Restart:       $COMPOSE_CMD restart"
echo ""
echo -e "${BLUE}Watching logs (Ctrl+C to exit)...${NC}"
echo ""

# Follow logs
$COMPOSE_CMD logs -f
