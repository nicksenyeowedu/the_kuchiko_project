# Dockerfile for Kuchiko Telegram Chatbot

# ---- Single stage: all deps have pre-built wheels, no compiler needed ----
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (all have pre-built wheels for linux/amd64)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY chatbot_telegram.py .
COPY build_embeddings.py .
COPY createKG.py .
COPY token_tracker.py .

# Note: The following files are NOT copied into the image:
# - export_chat_history.py (optional utility, not needed in container)
# - delete_chat_history.py (optional utility, not needed in container)
# - faiss_index.bin, entities.pkl, entity_metadata.pkl (generated at runtime by init container)
# - kg.pdf (mounted via volume)

# Create directory for logs
RUN mkdir -p /app/logs

# Set environment variables (will be overridden by docker-compose)
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for health checks)
EXPOSE 8000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the bot
CMD ["python", "-u", "chatbot_telegram.py"]
