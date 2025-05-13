FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone LightRAG repository
RUN git clone https://github.com/HKUDS/LightRAG.git .

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Set environment variables
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV HOST=0.0.0.0
ENV PORT=9621

# Expose the default port
EXPOSE 9621

# Create the API script inline to avoid Python detection
RUN cat > /app/lightrag_api.py << 'EOF'
import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_embedding, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

# [Rest of the Python code goes here - copy from the previous artifact]
EOF

CMD ["python", "lightrag_api.py"]