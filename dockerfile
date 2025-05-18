FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone LightRAG repository
RUN git clone https://github.com/HKUDS/LightRAG.git /tmp/lightrag

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/lightrag/requirements.txt && \
    pip install --no-cache-dir /tmp/lightrag[api] && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart aiofiles wcwidth

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Set environment variables
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV HOST=0.0.0.0
ENV PORT=9621

# Create .env file
RUN echo "WORKING_DIR=/app/data/rag_storage\n\
INPUT_DIR=/app/data/inputs\n\
HOST=0.0.0.0\n\
PORT=9621\n\
EMBEDDING_BINDING=openai\n\
EMBEDDING_MODEL=text-embedding-ada-002\n\
LLM_BINDING=openai\n\
LLM_MODEL=gpt-4" > /app/.env

# Create a simple script to start the server
RUN echo '#!/bin/bash\n\
cd /app\n\
exec lightrag-server\n' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose the port
EXPOSE 9621

# Run the server
CMD ["/app/start.sh"]