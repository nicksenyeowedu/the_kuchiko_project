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

# Detect OS and distribution
detect_os() {
    OS="unknown"
    DISTRO="unknown"
    PKG_MANAGER="unknown"

    case "$(uname -s)" in
        Linux*)
            OS="linux"
            # Detect Linux distribution
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                DISTRO="$ID"
            elif [ -f /etc/redhat-release ]; then
                DISTRO="rhel"
            elif [ -f /etc/debian_version ]; then
                DISTRO="debian"
            fi

            # Set package manager based on distro
            case "$DISTRO" in
                ubuntu|debian|linuxmint|pop)
                    PKG_MANAGER="apt"
                    ;;
                fedora)
                    PKG_MANAGER="dnf"
                    ;;
                centos|rhel|rocky|almalinux)
                    if command -v dnf &> /dev/null; then
                        PKG_MANAGER="dnf"
                    else
                        PKG_MANAGER="yum"
                    fi
                    ;;
                arch|manjaro|endeavouros)
                    PKG_MANAGER="pacman"
                    ;;
                opensuse*|sles)
                    PKG_MANAGER="zypper"
                    ;;
                alpine)
                    PKG_MANAGER="apk"
                    ;;
            esac
            ;;
        Darwin*)
            OS="macos"
            DISTRO="macos"
            if command -v brew &> /dev/null; then
                PKG_MANAGER="brew"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            OS="windows"
            DISTRO="windows"
            ;;
    esac

    # Check for WSL
    if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
        OS="wsl"
    fi

    echo -e "${BLUE}Detected OS: $OS ($DISTRO) - Package Manager: $PKG_MANAGER${NC}"
}

# Install Docker based on OS/distro
install_docker() {
    echo -e "${YELLOW}Docker is not installed. Installing automatically...${NC}"

    case "$PKG_MANAGER" in
        apt)
            sudo apt-get update
            sudo apt-get install -y docker.io docker-compose
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        dnf)
            sudo dnf -y install dnf-plugins-core
            sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
            sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        yum)
            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        pacman)
            sudo pacman -Sy --noconfirm docker docker-compose
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        zypper)
            sudo zypper install -y docker docker-compose
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        apk)
            sudo apk add docker docker-compose
            sudo rc-update add docker boot
            sudo service docker start
            sudo addgroup $USER docker
            ;;
        brew)
            echo -e "${YELLOW}Installing Docker Desktop via Homebrew...${NC}"
            brew install --cask docker
            echo -e "${YELLOW}Please open Docker Desktop from Applications to complete setup.${NC}"
            echo -e "${YELLOW}Then run this script again.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unsupported system: $DISTRO (package manager: $PKG_MANAGER)${NC}"
            echo "Please install Docker manually from https://docs.docker.com/get-docker/"
            exit 1
            ;;
    esac

    echo -e "${GREEN}✓ Docker installed successfully${NC}"
    echo -e "${YELLOW}Note: You may need to log out and back in for group changes to take effect.${NC}"
}

# Install Docker Compose based on OS/distro
install_docker_compose() {
    echo -e "${YELLOW}Docker Compose not found. Installing...${NC}"
    case "$PKG_MANAGER" in
        apt)
            sudo apt-get install -y docker-compose
            ;;
        dnf|yum)
            # Already installed with docker-compose-plugin
            ;;
        pacman)
            sudo pacman -Sy --noconfirm docker-compose
            ;;
        zypper)
            sudo zypper install -y docker-compose
            ;;
        apk)
            sudo apk add docker-compose
            ;;
        *)
            echo -e "${RED}❌ Please install Docker Compose manually${NC}"
            exit 1
            ;;
    esac
    echo -e "${GREEN}✓ Docker Compose installed${NC}"
}

# Start Docker daemon based on OS
start_docker_daemon() {
    echo -e "${YELLOW}Docker daemon is not running. Starting...${NC}"
    case "$PKG_MANAGER" in
        apt|dnf|yum|pacman|zypper)
            sudo systemctl start docker
            ;;
        apk)
            sudo service docker start
            ;;
        brew)
            echo -e "${YELLOW}Please open Docker Desktop from Applications${NC}"
            exit 1
            ;;
        *)
            echo -e "${RED}❌ Please start Docker manually${NC}"
            exit 1
            ;;
    esac
}

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
    echo "  nano .env"
    echo ""
    echo "Add the following content:"
    echo "  NVIDIA_API_KEY=your_nvidia_api_key_here"
    echo "  TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here"
    echo "  PDF_FILE=kg.pdf"
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

# Step 3: Check and install Docker if needed
echo -e "${YELLOW}[3/5] Checking Docker installation...${NC}"

# Detect OS first
detect_os

if ! command -v docker &> /dev/null; then
    install_docker
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    install_docker_compose
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

# Step 4: Check if Docker daemon is running
echo -e "${YELLOW}[4/5] Checking Docker daemon...${NC}"
# Try without sudo first, then with sudo
if ! docker info &> /dev/null && ! sudo docker info &> /dev/null; then
    start_docker_daemon
    sleep 2
    if ! sudo docker info &> /dev/null; then
        echo -e "${RED}❌ Failed to start Docker daemon${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker daemon started${NC}"
fi
echo -e "${GREEN}✓ Docker daemon is running${NC}"

# Step 5: Start the services
echo -e "${YELLOW}[5/5] Starting Kuchiko services...${NC}"
echo ""

# Determine if we need sudo for docker commands
SUDO_CMD=""
if ! docker ps &> /dev/null; then
    SUDO_CMD="sudo"
fi

# Determine docker-compose command
if $SUDO_CMD docker compose version &> /dev/null; then
    COMPOSE_CMD="$SUDO_CMD docker compose"
else
    COMPOSE_CMD="$SUDO_CMD docker-compose"
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
