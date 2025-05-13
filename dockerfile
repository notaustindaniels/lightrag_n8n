
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone LightRAG repository
RUN git clone https://github.com/HKUDS/LightRAG.git /tmp/lightrag

# Copy necessary files from LightRAG
RUN cp -r /tmp/lightrag/lightrag /app/lightrag && \
    cp /tmp/lightrag/setup.py /app/setup.py && \
    cp /tmp/lightrag/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .
RUN pip install fastapi uvicorn[standard] python-multipart aiofiles

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Set environment variables
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV HOST=0.0.0.0
ENV PORT=9621

# Copy the API script
COPY lightrag_api.py /app/lightrag_api.py

# Expose the default port
EXPOSE 9621

CMD ["python", "lightrag_api.py"]