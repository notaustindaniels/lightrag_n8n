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

# Copy the extended API and migration script
COPY lightrag_extended_api.py /app/
COPY migrate_metadata.py /app/

# Create webui directory
RUN mkdir -p /app/webui

# Copy webui directory if it exists (using a two-stage approach)
# First, copy everything to ensure build doesn't fail
COPY . /tmp/build/
# Then selectively copy webui if it exists
RUN if [ -d "/tmp/build/webui" ]; then cp -r /tmp/build/webui/* /app/webui/; fi && \
    rm -rf /tmp/build

# Check if webui exists in the installed package and log the result
RUN python -c "import site; import glob; import os; \
    found = False; \
    for site_dir in site.getsitepackages(): \
        pattern = os.path.join(site_dir, 'lightrag*/lightrag/api/webui'); \
        matches = glob.glob(pattern); \
        if matches and os.path.exists(matches[0]): \
            print(f'Package WebUI found at: {matches[0]}'); \
            found = True; \
            break; \
    if not found: print('Package WebUI not found'); \
    print(f'Local WebUI fallback will be at: /app/webui')"

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

# Run the extended API server
CMD ["python", "/app/lightrag_extended_api.py"]