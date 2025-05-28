FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages including lightrag with API support
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir lightrag-hku[api] && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart aiofiles wcwidth pydantic

# Clone LightRAG to ensure we have the web UI files
RUN git clone https://github.com/HKUDS/LightRAG.git /tmp/lightrag && \
    if [ -d "/tmp/lightrag/lightrag/api/webui" ]; then \
        cp -r /tmp/lightrag/lightrag/api/webui /app/webui; \
    fi

# Copy the extended API and migration script
COPY lightrag_extended_api.py /app/
COPY migrate_metadata.py /app/

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Set environment variables
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV HOST=0.0.0.0
ENV PORT=9621
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 9621

# Run the extended API server which includes the web UI
CMD ["python", "/app/lightrag_extended_api.py"]