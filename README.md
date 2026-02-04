# Kuchiko - Knowledge Graph Telegram Chatbot

A Telegram chatbot that answers questions using a knowledge graph built from your PDF documents. Powered by NVIDIA NIM, Memgraph, and a hybrid RAG (Retrieval-Augmented Generation) architecture.

## Features

- **PDF to Knowledge Graph**: Automatically extracts entities and relationships from your PDF
- **Hybrid RAG Search**: Combines vector search (FAISS) with graph traversal for accurate answers
- **Telegram Interface**: Easy-to-use chat interface via Telegram
- **Conversation Memory**: Remembers context within conversations
- **Multi-user Support**: Handles multiple users simultaneously
- **Auto-Setup**: Automatically detects your OS and installs Docker if needed

## Prerequisites

Before starting, you only need:

- A **NVIDIA NIM API key** (free tier available)
- A **Telegram Bot Token**

**Note:** Docker will be automatically installed by the setup script if not present.

---

## Supported Platforms

The setup script automatically detects your operating system and installs Docker using the appropriate package manager:

| OS | Distribution | Package Manager | Auto-Install |
|----|--------------|-----------------|--------------|
| Linux | Ubuntu, Debian, Linux Mint, Pop!_OS | apt | ✅ Yes |
| Linux | Fedora | dnf | ✅ Yes |
| Linux | CentOS, RHEL, Rocky, AlmaLinux | dnf/yum | ✅ Yes |
| Linux | Arch, Manjaro, EndeavourOS | pacman | ✅ Yes |
| Linux | openSUSE, SLES | zypper | ✅ Yes |
| Linux | Alpine | apk | ✅ Yes |
| macOS | All versions | Homebrew | ✅ Yes (requires manual Docker Desktop launch) |
| Windows | WSL2 | Uses Linux method | ✅ Yes |
| Windows | Native | winget/Chocolatey | ✅ Yes (requires Docker Desktop launch) |

---

## Quick Start

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/hybrid_kg_rag.git
cd hybrid_kg_rag
```

### Step 2: Add your PDF file

Place your knowledge base PDF in the project folder. Name it `kg.pdf` or use a custom name (update `.env` accordingly).

### Step 3: Create your environment file

```bash
nano .env
```

Add the following content (replace with your actual keys):
```env
NVIDIA_API_KEY=nvapi-your-actual-key-here
TELEGRAM_BOT_TOKEN=your-telegram-bot-token-here
PDF_FILE=kg.pdf
```

### Step 4: Run the setup script

**Linux/macOS:**
```bash
bash start.sh
```

**Windows (PowerShell):**
```powershell
.\start.ps1
```

**Windows (Command Prompt):**
```cmd
start.bat
```

That's it! The script will:
1. Detect your operating system
2. Install Docker automatically (if not installed)
3. Start the Docker daemon
4. Build and launch all services
5. Show you the logs

**First time running?** If Docker was just installed, you may need to:
- **Linux:** Run `newgrp docker` or log out/in for permissions
- **Windows/macOS:** Start Docker Desktop manually, then run the script again

See [Troubleshooting](#troubleshooting) if you encounter issues.

---

## Getting Your API Keys

### NVIDIA NIM API Key

1. Go to [NVIDIA Build](https://build.nvidia.com/)
2. Sign in or create a free account
3. Navigate to any model (e.g., DeepSeek)
4. Click "Get API Key" or find it in your account settings
5. Copy your API key (starts with `nvapi-`)

### Telegram Bot Token

1. Open Telegram and search for **@BotFather**
2. Start a chat and send `/newbot`
3. Follow the prompts:
   - Enter a **name** for your bot (e.g., "My Knowledge Bot")
   - Enter a **username** for your bot (must end in `bot`, e.g., "my_knowledge_bot")
4. BotFather will give you a **token** - copy it (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

---

## What the Setup Script Does

When you run `bash start.sh`, it performs these steps:

```
[1/5] Checking environment file...     → Validates .env exists
[2/5] Validating environment variables... → Checks API keys are set
[3/5] Checking Docker installation...  → Auto-installs Docker if missing
[4/5] Checking Docker daemon...        → Starts Docker if not running
[5/5] Starting Kuchiko services...     → Builds and launches containers
```

**Behind the scenes:**
1. Memgraph database starts
2. Your PDF is processed into a knowledge graph
3. FAISS embeddings are built for fast search
4. The Telegram bot starts and connects

---

## First Time Setup - What to Expect

When you run `bash start.sh` for the first time on a fresh system:

1. **Docker Installation** (if not present)
   - The script detects your OS and installs Docker automatically
   - You'll be prompted for your sudo password
   - Your user is added to the `docker` group

2. **Permission Note**
   - After Docker installs, you may need to run `newgrp docker` or log out/in
   - The script attempts to handle this automatically with sudo

3. **Container Build**
   - Docker images are downloaded and built
   - This can take several minutes on first run

4. **Knowledge Graph Creation**
   - Your PDF is processed and converted to a knowledge graph
   - Embeddings are generated for semantic search

5. **Bot Startup**
   - The Telegram bot connects and starts listening
   - You'll see logs streaming in your terminal

**Expected output on success:**
```
✅ Kuchiko is starting up!
The bot is initializing...
Watching logs (Ctrl+C to exit)...
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

## Managing the Bot

### Start the bot

**Linux/macOS:**
```bash
bash start.sh
```

**Windows:**
```powershell
.\start.ps1
```

### Stop the bot

**Linux/macOS:**
```bash
bash stop.sh
```

**Windows:**
```powershell
.\stop.ps1
```

**Any platform:**
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs -f
```

### Restart
```bash
docker-compose restart
```

### Rebuild from scratch

**Linux/macOS:**
```bash
docker-compose down -v    # Remove all data
bash start.sh             # Rebuild everything
```

**Windows:**
```powershell
docker-compose down -v    # Remove all data
.\start.ps1               # Rebuild everything
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NVIDIA_API_KEY` | Yes | - | Your NVIDIA NIM API key |
| `TELEGRAM_BOT_TOKEN` | Yes | - | Your Telegram bot token |
| `PDF_FILE` | No | `kg.pdf` | Name of your PDF file |
| `MEMGRAPH_URI` | No | `bolt://localhost:7687` | Memgraph connection URI |
| `MEMGRAPH_USER` | No | `memgraph` | Memgraph username |
| `MEMGRAPH_PASS` | No | `memgraph` | Memgraph password |

### Changing the PDF File

1. Place your new PDF file in the project root
2. Update `.env`:
   ```env
   PDF_FILE=my_new_document.pdf
   ```
3. Rebuild the knowledge graph:
   ```bash
   docker-compose down -v
   bash start.sh
   ```

---

## Troubleshooting

### Docker permission denied (most common on first run)

After Docker is freshly installed, your user is added to the `docker` group but the change doesn't take effect until you log out and back in. You have three options:

**Option 1: Apply group change immediately (recommended)**
```bash
newgrp docker
bash start.sh
```

**Option 2: Log out and log back in**
```bash
# Log out of your SSH session or terminal
exit
# Log back in, then run
bash start.sh
```

**Option 3: Run with sudo**
```bash
sudo bash start.sh
```

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

### "Failed to start Docker daemon"

If you see this error, Docker may need sudo to start:
```bash
sudo systemctl start docker
bash start.sh
```

### Rate limit errors
The NVIDIA NIM free tier has rate limits. If you see "429 Too Many Requests" errors, wait a few minutes and try again.

### macOS: Docker Desktop not starting
On macOS, after Homebrew installs Docker Desktop, you need to manually open it from Applications once. Then run the script again.

### Windows: Docker Desktop not running

If you see "Docker daemon is not running":
1. Open Docker Desktop from the Start Menu
2. Wait for it to fully start (check the system tray icon)
3. Run the script again

### Windows: PowerShell execution policy error

If you see "running scripts is disabled on this system":
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run the script with bypass:
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

### Windows: winget or Chocolatey not found

If Docker can't auto-install, install it manually:
1. Download from https://www.docker.com/products/docker-desktop/
2. Run the installer
3. Start Docker Desktop
4. Run the script again

### Script says "command not found" or "permission denied"

Make sure you're running from the project directory:
```bash
cd hybrid_kg_rag
bash start.sh
```

Or make the script executable:
```bash
chmod +x start.sh
./start.sh
```

---

## Project Structure

```
kuchiko/
├── start.sh                # Linux/macOS setup script (auto-installs Docker)
├── start.ps1               # Windows PowerShell setup script
├── start.bat               # Windows batch file (runs start.ps1)
├── stop.sh                 # Stop the bot (Linux/macOS)
├── stop.ps1                # Stop the bot (Windows)
├── .env                    # Your environment variables
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

## Technical Documentation

For a detailed explanation of the entire pipeline architecture, including:

- PDF extraction and semantic chunking
- Knowledge graph creation with LLM entity extraction
- Embedding generation and FAISS index building
- Hybrid RAG retrieval (Vector Search + Graph Expansion)
- Response generation pipeline

See **[ARCHITECTURE.md](ARCHITECTURE.md)**

---

## License

MIT License

---

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the logs: `docker-compose logs -f`
3. Open an issue on GitHub
