#!/usr/bin/env python3
"""
Simple delete service that runs alongside LightRAG
"""
import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

app = FastAPI(title="LightRAG Delete Service")

class DeleteDocumentRequest(BaseModel):
    doc_ids: List[str]

@app.delete("/documents/by-id")
async def delete_documents_by_id(request: DeleteDocumentRequest):
    """Delete documents by their IDs"""
    try:
        rag = LightRAG(
            working_dir=os.getenv("WORKING_DIR", "/app/data/rag_storage"),
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )
        
        rag.delete_by_doc_id(request.doc_ids)
        
        return {
            "status": "success",
            "message": f"Successfully deleted {len(request.doc_ids)} documents",
            "deleted_ids": request.doc_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9622)