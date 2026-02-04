# Kuchiko - Knowledge Graph Telegram Chatbot

A Telegram chatbot that answers questions using a knowledge graph built from your PDF documents. Powered by NVIDIA NIM, Memgraph, and a hybrid RAG (Retrieval-Augmented Generation) architecture.

## Features

- **PDF to Knowledge Graph**: Automatically extracts entities and relationships from your PDF
- **Hybrid RAG Search**: Combines vector search (FAISS) with graph traversal for accurate answers
- **Telegram Interface**: Easy-to-use chat interface via Telegram
- **Conversation Memory**: Remembers context within conversations
- **Multi-user Support**: Handles multiple users simultaneously

## Prerequisites

Before starting, make sure you have:

- **Docker** and **Docker Compose** installed on your system
- A **NVIDIA NIM API key** (free tier available)
- A **Telegram Bot Token**

---

## Quick Start (3 Steps)

```bash
# 1. Create your environment file
cp .env.example .env

# 2. Edit .env with your API keys (see setup guide below)
nano .env

# 3. Run the setup script
./start.sh
```

That's it! The script handles everything else automatically.

---

## Detailed Setup Guide

### Step 1: Install Docker

#### macOS
```bash
# Using Homebrew
brew install --cask docker

# Then open Docker Desktop from Applications
```

#### Windows
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
2. Run the installer
3. Start Docker Desktop

#### Linux (Ubuntu/Debian)
```bash
# Update package index
sudo apt-get update

# Install Docker
sudo apt-get install docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (optional, avoids using sudo)
sudo usermod -aG docker $USER
```

Verify Docker is installed:
```bash
docker --version
docker-compose --version
```

---

### Step 2: Get Your NVIDIA NIM API Key

1. Go to [NVIDIA Build](https://build.nvidia.com/)
2. Sign in or create a free account
3. Navigate to any model (e.g., DeepSeek)
4. Click "Get API Key" or find it in your account settings
5. Copy your API key (starts with `nvapi-`)

---

### Step 3: Create Your Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Start a chat and send `/newbot`
3. Follow the prompts:
   - Enter a **name** for your bot (e.g., "My Knowledge Bot")
   - Enter a **username** for your bot (must end in `bot`, e.g., "my_knowledge_bot")
4. BotFather will give you a **token** - copy it (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

---

### Step 4: Set Up the Project

1. **Clone or download this repository**

2. **Add your PDF file**

   Place your knowledge base PDF in the project root folder. By default, it should be named `kg.pdf`, but you can use any name.

3. **Create your environment file**

   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

4. **Edit the `.env` file** with your credentials:
   ```bash
   # Open with your preferred editor
   nano .env
   # or
   code .env
   ```

   Fill in your values:
   ```env
   # REQUIRED
   NVIDIA_API_KEY=nvapi-your-actual-key-here
   TELEGRAM_BOT_TOKEN=your-telegram-bot-token-here

   # OPTIONAL: If your PDF has a different name
   PDF_FILE=your_document.pdf
   ```

---

### Step 5: Run the Bot

Simply run the start script:

```bash
./start.sh
```

The script will:
1. Validate your `.env` file
2. Check Docker is installed and running
3. Start all services automatically
4. Show you the logs

**What happens behind the scenes:**
1. Memgraph database starts
2. Your PDF is processed into a knowledge graph (takes a few minutes on first run)
3. FAISS embeddings are built for fast search
4. The Telegram bot starts and connects

**Other commands:**
```bash
# Stop the bot
./stop.sh

# View logs manually
docker-compose logs -f

# Restart
docker-compose restart
```

---

## Usage

Once running, open Telegram and start chatting with your bot:

### Commands

| Command | Description |
|---------|-------------|
| `/start` | Start conversation with Kuchiko |
| `/help` | Show help message |
| `/table` | View knowledge graph table of contents |
| `/random` | Discover a random topic |
| `/reset` | Clear your chat history |

### Example Questions

- "Who is James Brooke?"
- "Tell me about the history of Kuching"
- "What happened during the Japanese occupation?"

---

## Configuration Options

### Changing the PDF File

1. Place your new PDF file in the project root
2. Update `.env`:
   ```env
   PDF_FILE=my_new_document.pdf
   ```
3. Rebuild the knowledge graph:
   ```bash
   docker-compose down -v  # Remove old data
   docker-compose up -d    # Rebuild with new PDF
   ```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NVIDIA_API_KEY` | Yes | - | Your NVIDIA NIM API key |
| `TELEGRAM_BOT_TOKEN` | Yes | - | Your Telegram bot token |
| `PDF_FILE` | No | `kg.pdf` | Name of your PDF file |
| `MEMGRAPH_URI` | No | `bolt://localhost:7687` | Memgraph connection URI |
| `MEMGRAPH_USER` | No | `memgraph` | Memgraph username |
| `MEMGRAPH_PASS` | No | `memgraph` | Memgraph password |

---

## Troubleshooting

### Bot not responding
```bash
# Check if containers are running
docker-compose ps

# Check chatbot logs for errors
docker-compose logs chatbot
```

### Knowledge graph not building
```bash
# Check init container logs
docker-compose logs init-kg
```

### Rate limit errors
The NVIDIA NIM free tier has rate limits. If you see "429 Too Many Requests" errors, wait a few minutes and try again.

### Rebuild everything from scratch
```bash
# Stop and remove all containers and volumes
docker-compose down -v

# Rebuild and start
docker-compose up -d --build
```

---

## Project Structure

```
kuchiko/
├── start.sh                # 🚀 Run this to start everything
├── stop.sh                 # Stop the bot
├── .env                    # Your environment variables (create from .env.example)
├── .env.example            # Template for environment variables
├── .gitignore              # Git ignore rules
├── docker-compose.yml      # Docker services configuration
├── Dockerfile              # Container build instructions
├── requirements.txt        # Python dependencies
├── kg.pdf                  # Your knowledge base PDF (or custom name)
├── chatbot_telegram.py     # Main Telegram bot code
├── createKG.py             # PDF to Knowledge Graph processor
├── build_embeddings.py     # FAISS index builder
├── data/                   # Generated data (FAISS index, embeddings)
└── logs/                   # Application logs
```

---

## License

MIT License

---

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the logs: `docker-compose logs -f`
3. Open an issue on GitHub
