# Kuchiko - Knowledge Graph Telegram Chatbot

A Telegram chatbot that answers questions using a knowledge graph built from your PDF documents. Powered by NVIDIA NIM, Memgraph, and a hybrid RAG (Retrieval-Augmented Generation) architecture.

## Features

- **PDF to Knowledge Graph**: Automatically extracts entities and relationships from your PDF
- **Hybrid RAG Search**: Combines vector search (FAISS) with graph traversal for accurate answers
- **Telegram Interface**: Easy-to-use chat interface via Telegram
- **Conversation Memory**: Remembers context within conversations
- **Multi-user Support**: Handles multiple users simultaneously

## Prerequisites

Before starting, you need:

- A **NVIDIA NIM API key** (free tier available)
- A **Telegram Bot Token**
- **Docker** installed and running on your machine

### Installing Docker

Download and install Docker from the official website: **[https://www.docker.com/get-started/](https://www.docker.com/get-started/)**

> **Important:**
> - **Windows & macOS**: You must install Docker Desktop and **start it manually** before running the setup script.
> - **Linux**: Docker installation and startup is handled automatically by the setup script.

---

## Supported Platforms

| OS | Distribution | Docker Setup |
|----|--------------|--------------|
| Linux | Ubuntu, Debian, Linux Mint, Pop!_OS | Automatic (apt) |
| Linux | Fedora | Automatic (dnf) |
| Linux | CentOS, RHEL, Rocky, AlmaLinux | Automatic (dnf/yum) |
| Linux | Arch, Manjaro, EndeavourOS | Automatic (pacman) |
| Linux | openSUSE, SLES | Automatic (zypper) |
| Linux | Alpine | Automatic (apk) |
| macOS | All versions | [Install manually](https://www.docker.com/get-started/) — start Docker Desktop before running the script |
| Windows | WSL2 / Native | [Install manually](https://www.docker.com/get-started/) — start Docker Desktop before running the script |

---

## Quick Start

### Step 1: Clone the repository

```bash
git clone https://github.com/nicksenyeowedu/the_kuchiko_project.git
cd the_kuchiko_project
```

### Step 2: Add your PDF file

Place your knowledge base PDF in the project folder. Name it `kg.pdf` or use a custom name (update `.env` accordingly).

### Step 3: Configure your environment file

A `.env` file is already included in the project. Open it and fill in your actual API keys:

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
2. On Linux: install Docker automatically if not installed
3. Start the Docker daemon (Linux only — on Windows/macOS, make sure Docker Desktop is already running)
4. Build and launch all services
5. Show you the logs

**Viewing build progress:**
- **Windows:** Build progress is shown directly in the terminal.
- **Linux/macOS:** The script runs in detached mode. To see build progress, open a **second terminal** and run:
  ```bash
  docker-compose logs -f
  ```

**First time running?**
- **Linux:** If Docker was just installed, you may need to run `newgrp docker` or log out/in for permissions
- **Windows/macOS:** Make sure Docker Desktop is running before executing the script

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
[3/5] Checking Docker installation...  → Auto-installs Docker if missing (Linux only)
[4/5] Checking Docker daemon...        → Starts Docker if not running (Linux only)
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

1. **Docker Setup**
   - **Linux:** The script detects your distro and installs Docker automatically. You'll be prompted for your sudo password and your user is added to the `docker` group.
   - **Windows/macOS:** Make sure you have [Docker Desktop](https://www.docker.com/get-started/) installed and running before proceeding.

2. **Permission Note (Linux only)**
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

## Common Commands

| Action | Linux/macOS | Windows (PowerShell) |
|--------|-------------|----------------------|
| **Start** | `bash start.sh` | `.\start.ps1` |
| **Stop** | `bash stop.sh` | `.\stop.ps1` |
| **Stop (any platform)** | `docker-compose down` | `docker-compose down` |
| **View logs** | `docker-compose logs -f` | `docker-compose logs -f` |
| **Restart** | `docker-compose restart` | `docker-compose restart` |

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

### Rebuild the knowledge graph only
```bash
docker compose up --build init-kg
```

### Full Docker cleanup (advanced — removes all Docker data)
```bash
docker system prune -a
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
| `MAX_WORKERS` | No | `4` | Number of parallel workers for KG processing |

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

### macOS/Windows: Docker Desktop not running

If you see "Docker daemon is not running":
1. Open Docker Desktop (from Applications on macOS, or Start Menu on Windows)
2. Wait for it to fully start (check the system tray/menu bar icon)
3. Run the script again

> Docker Desktop must be running before you execute the setup script on Windows and macOS. Download it from [https://www.docker.com/get-started/](https://www.docker.com/get-started/) if you haven't already.

### Windows: PowerShell execution policy error

If you see "running scripts is disabled on this system":
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run the script with bypass:
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

### Script says "command not found" or "permission denied"

Make sure you're running from the project directory:
```bash
cd the_kuchiko_project
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
├── start.sh                # Linux/macOS setup script
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

## Models Used

This project uses three models, each serving a different role in the pipeline:

### 1. DeepSeek V3.1 — LLM for Entity & Relationship Extraction

| | |
|---|---|
| **Model** | [deepseek-ai/deepseek-v3.1](https://build.nvidia.com/deepseek-ai/deepseek-v3_1) |
| **Provider** | NVIDIA NIM (free tier) |
| **Parameters** | 671B total (37B activated via MoE) |
| **Context Window** | 128,000 tokens |
| **Architecture** | Transformer (decoder-only, Mixture of Experts) |
| **License** | MIT |

**Used for:** Extracting entities and relationships from PDF text (`createKG.py`), refining semantic section boundaries, and generating chatbot responses (`chatbot_telegram.py`).

### 2. Llama 3.2 NeMo Retriever 300M Embed V2 — Text Embedding API

| | |
|---|---|
| **Model** | [nvidia/llama-3.2-nemoretriever-300m-embed-v2](https://build.nvidia.com/nvidia/llama-3_2-nemoretriever-300m-embed-v2) |
| **Provider** | NVIDIA NIM (free tier) |
| **Parameters** | 300M |
| **Embedding Dimension** | 2048 (configurable: 384, 512, 768, 1024, 2048) |
| **Max Input Length** | 8,192 tokens |
| **Architecture** | Transformer (encoder, 9 layers) |
| **License** | NVIDIA Community Model License |

**Used for:** Generating vector embeddings for FAISS index (`build_embeddings.py`). These embeddings power the vector search component of the hybrid RAG retrieval.

### 3. all-MiniLM-L6-v2 — Local Embedding Model

| | |
|---|---|
| **Model** | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Provider** | Local (runs on CPU, no API needed) |
| **Parameters** | 22.7M |
| **Embedding Dimension** | 384 |
| **Max Input Length** | 256 word pieces |
| **Architecture** | MiniLM (distilled transformer) |
| **License** | Apache 2.0 |

**Used for:** Entity deduplication and page clustering during knowledge graph creation (`createKG.py`). Runs locally with no API cost.

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

## TODO

- [x] PDF extraction and semantic chunking
- [x] Knowledge graph creation with LLM entity extraction
- [x] Embedding-based entity deduplication
- [x] FAISS vector index for fast search
- [x] Hybrid RAG retrieval (vector search + graph traversal)
- [x] Telegram bot interface with conversation memory
- [x] Parallel processing with configurable workers
- [x] Cross-platform support (Windows, macOS, Linux)
- [x] Automated Docker setup for Linux
- [x] Retry logic for Memgraph transaction conflicts
- [x] Run build experiments
- [x] Calculate average token usage across multiple runs
- [ ] Estimated cost comparison for alternative LLMs
- [ ] Option to use other LLM providers (Eg: Anthropic, OpenAI, Gemini, etc.)


---

## Build Statistics

Build statistics from running the full pipeline (PDF extraction, KG creation, FAISS index).

### Test Configuration

| Variable | Value |
|----------|-------|
| `MAX_WORKERS` | 20 |
| PDF pages | 114 |
| PDF word count | ~25,900 words |

### Token Usage (Per Experiment)

Results from 10 experiments across macOS and Windows.

| Exp | LLM Calls | LLM Tokens | Embed Calls | Embed Tokens | Local Embed Calls | Total API Tokens | Build Time |
|-----|-----------|------------|-------------|--------------|-------------------|-----------------|------------|
| 1 | 650 | 619,864 | 105 | 180,471 | 5,468 | 800,335 | 7m 21s |
| 2 | 631 | 642,228 | 93 | 155,736 | 5,483 | 797,964 | 6m 16s |
| 3 | 638 | 623,038 | 98 | 161,025 | 5,673 | 784,063 | 6m 3s |
| 4 | 628 | 619,170 | 97 | 158,958 | 5,570 | 778,128 | 7m 16s |
| 5 | 610 | 615,709 | 99 | 163,304 | 5,713 | 779,013 | 6m 39s |
| 6 | 638 | 630,910 | 100 | 165,324 | 5,781 | 796,234 | 8m 6s |
| 7 | 613 | 597,661 | 91 | 151,896 | 5,255 | 749,557 | 7m 7s |
| 8 | 628 | 623,414 | 98 | 160,641 | 5,657 | 784,055 | 7m 53s |
| 9 | 624 | 612,289 | 97 | 160,007 | 5,564 | 772,296 | 10m 10s |
| 10 | 629 | 608,357 | 93 | 154,346 | 5,418 | 762,697 | 10m 35s |

### LLM Token Breakdown (Input vs Output)

| Exp | LLM Calls | Input Tokens | Output Tokens | Total LLM Tokens |
|-----|-----------|-------------|---------------|-----------------|
| 1 | 650 | 300,409 | 319,455 | 619,864 |
| 2 | 631 | 292,613 | 349,615 | 642,228 |
| 3 | 638 | 295,570 | 327,468 | 623,038 |
| 4 | 628 | 291,625 | 327,545 | 619,170 |
| 5 | 610 | 284,300 | 331,409 | 615,709 |
| 6 | 638 | 295,552 | 335,358 | 630,910 |
| 7 | 613 | 285,501 | 312,160 | 597,661 |
| 8 | 628 | 291,705 | 331,709 | 623,414 |
| 9 | 624 | 290,045 | 322,244 | 612,289 |
| 10 | 629 | 291,827 | 316,524 | 608,351 |
| **Avg** | **629** | **291,915** | **327,349** | **619,264** |

### Average Token Usage (All Experiments)

| Component | Avg API Calls | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens |
|-----------|---------------|-----------------|-------------------|-----------------|
| LLM Chat Completions (NVIDIA NIM) | 629 | 291,915 | 327,349 | 619,264 |
| Embedding API (NVIDIA NIM) | 97 | 161,171 | — | 161,171 |
| Local Embeddings (sentence-transformers) | 5,558 | N/A | N/A | N/A (no API cost) |
| **Grand Total** | | | | **780,434** |

### Build Time (All Experiments)

| Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Exp 7 | Exp 8 | Exp 9 | Exp 10 | **Average** |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-------------|
| 7m 21s | 6m 16s | 6m 3s | 7m 16s | 6m 39s | 8m 6s | 7m 7s | 7m 53s | 10m 10s | 10m 35s | **7m 45s** |

---

## Estimated Cost (Alternative LLMs)

If you were to use a paid LLM instead of the NVIDIA NIM free tier, the estimated cost per build would be:

| Provider | Model | Input Cost | Output Cost | Estimated Total |
|----------|-------|------------|-------------|-----------------|
| NVIDIA NIM | DeepSeek V3.1 | Free | Free | **$0.00** |
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> Costs calculated based on average token usage of ~780,000 total API tokens per build (~619,000 LLM + ~161,000 embedding).

---

## License

MIT License

---

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the logs: `docker-compose logs -f`
3. Open an issue on GitHub





