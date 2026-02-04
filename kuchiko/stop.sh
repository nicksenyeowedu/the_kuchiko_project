#!/bin/bash

# Kuchiko - Stop Script

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BLUE}Stopping Kuchiko...${NC}"
echo ""

# Determine docker-compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

$COMPOSE_CMD down

echo ""
echo -e "${GREEN}✓ Kuchiko stopped${NC}"
echo ""
echo "To start again, run: ./start.sh"
echo ""
