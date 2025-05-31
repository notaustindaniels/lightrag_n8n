FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone LightRAG repository (for dependencies)
RUN git clone https://github.com/HKUDS/LightRAG.git /tmp/lightrag

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir lightrag-hku[api] && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart aiofiles wcwidth pydantic

# Copy scripts and API
COPY lightrag_extended_api.py /app/
COPY migrate_metadata.py /app/
COPY download_webui.sh /app/
COPY build_webui.sh /app/

# Make scripts executable
RUN chmod +x /app/download_webui.sh /app/build_webui.sh

# Try to get WebUI files
RUN /app/download_webui.sh || echo "WebUI download failed, continuing..."

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/webui

# Set environment variables
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV HOST=0.0.0.0
ENV PORT=9621
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 9621

# Run the extended API server
CMD ["python", "/app/lightrag_extended_api.py"]