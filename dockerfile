FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and install LightRAG with web UI support
RUN git clone https://github.com/HKUDS/LightRAG.git /tmp/lightrag && \
    cd /tmp/lightrag && \
    pip install --no-cache-dir -e ".[api,webui]" && \
    cd /app

# Copy requirements and install additional dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Set environment variables
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV HOST=0.0.0.0
ENV PORT=9621
ENV ENABLE_WEBUI=true

# Copy the API script
COPY lightrag_api.py /app/lightrag_api.py

# Expose the default port
EXPOSE 9621

CMD ["python", "-u", "lightrag_api.py"]