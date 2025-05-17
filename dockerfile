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

# Install dependencies from the repo
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/lightrag/requirements.txt && \
    pip install --no-cache-dir -e "/tmp/lightrag[api]" && \
    # Install additional dependencies
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart aiofiles wcwidth

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Set environment variables
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV HOST=0.0.0.0
ENV PORT=9621

# Create a simplified API script that uses the built-in LightRAG server
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
import uvicorn\n\
\n\
# Simplified script to run lightrag-server directly\n\
from lightrag.api.app import create_app\n\
\n\
if __name__ == "__main__":\n\
    # Get environment variables\n\
    host = os.getenv("HOST", "0.0.0.0")\n\
    port = int(os.getenv("PORT", "9621"))\n\
    log_level = os.getenv("LOG_LEVEL", "info").lower()\n\
    workers = int(os.getenv("WORKERS", "1"))\n\
    \n\
    app = create_app()\n\
    \n\
    # Start the server\n\
    if workers > 1:\n\
        uvicorn.run(\n\
            "lightrag.api.app:create_app",\n\
            host=host,\n\
            port=port,\n\
            log_level=log_level,\n\
            workers=workers,\n\
            factory=True,\n\
        )\n\
    else:\n\
        uvicorn.run(\n\
            app,\n\
            host=host,\n\
            port=port,\n\
            log_level=log_level,\n\
        )\n\
' > /app/run_server.py

# Make script executable
RUN chmod +x /app/run_server.py

# Expose the port
EXPOSE 9621

# Run the server
CMD ["python", "/app/run_server.py"]