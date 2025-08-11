#!/usr/bin/env python3
"""
Test script to diagnose LightRAG hybrid mode issues
"""
import asyncio
import sys
import json
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

async def test_query():
    # Initialize LightRAG with test workspace
    rag = LightRAG(
        working_dir='./rag_storage/workspaces/test/test',
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=openai_embed
        ),
    )

    # Test different query modes
    test_query = "What is LightRAG and how does it work?"
    
    print("="*60)
    print("Testing LightRAG Query Modes")
    print("="*60)
    
    modes = ["local", "global", "hybrid", "naive"]
    
    for mode in modes:
        print(f"\n### Testing {mode.upper()} mode ###")
        query_param = QueryParam(
            mode=mode,
            only_need_context=True,  # Only get context first
            top_k=10
        )
        
        context = await rag.aquery(test_query, param=query_param)
        
        print(f"\n{mode.upper()} Context Retrieved:")
        print("-" * 40)
        if context:
            # Show first 500 chars of context
            context_preview = str(context)[:500] + "..." if len(str(context)) > 500 else str(context)
            print(context_preview)
        else:
            print("No context retrieved!")
        
        # Now get the full response
        query_param.only_need_context = False
        response = await rag.aquery(test_query, param=query_param)
        
        print(f"\n{mode.upper()} Response:")
        print("-" * 40)
        # Show first 300 chars of response
        response_preview = str(response)[:300] + "..." if len(str(response)) > 300 else str(response)
        print(response_preview)
    
    # Check what's in the storage
    print("\n" + "="*60)
    print("Storage Check")
    print("="*60)
    
    # Check entities
    if hasattr(rag, 'entities_vdb') and rag.entities_vdb:
        try:
            entities = await rag.entities_vdb.query("LightRAG", top_k=5)
            print("\nTop 5 entities for 'LightRAG':")
            for i, entity in enumerate(entities):
                print(f"{i+1}. {entity}")
        except Exception as e:
            print(f"Error querying entities: {e}")
    
    # Check relationships
    if hasattr(rag, 'relationships_vdb') and rag.relationships_vdb:
        try:
            relationships = await rag.relationships_vdb.query("LightRAG", top_k=5)
            print("\nTop 5 relationships for 'LightRAG':")
            for i, rel in enumerate(relationships):
                print(f"{i+1}. {rel}")
        except Exception as e:
            print(f"Error querying relationships: {e}")
    
    # Check chunks
    if hasattr(rag, 'chunks_vdb') and rag.chunks_vdb:
        try:
            chunks = await rag.chunks_vdb.query("LightRAG", top_k=5)
            print("\nTop 5 chunks for 'LightRAG':")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        except Exception as e:
            print(f"Error querying chunks: {e}")

if __name__ == "__main__":
    asyncio.run(test_query())