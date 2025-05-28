# Enhanced LightRAG with Extended API

This repository extends the LightRAG REST API to solve the file_path validation error and add enhanced document management capabilities.

## Problem Solved

The original LightRAG `/documents/text` endpoint doesn't set a `file_path` field when inserting documents, which causes a Pydantic validation error when retrieving documents via the `/documents` endpoint. This enhanced version fixes this issue and adds additional functionality.

## Key Features

1. **Extended REST API** (`lightrag_extended_api.py`):
   - Fixes the file_path validation error by ensuring all documents have a file_path
   - Adds `/documents/text/enhanced` endpoint for rich metadata insertion
   - Adds `/documents/by-sitemap/{sitemap_identifier}` for sitemap-specific queries
   - Adds `/documents/by-sitemap/{sitemap_identifier}` DELETE endpoint for bulk deletion
   - Maintains an in-memory metadata store that persists to disk
   - Fully compatible with existing LightRAG Python API

2. **Enhanced n8n Workflow**:
   - Automatically deletes old documents from a sitemap before re-indexing
   - Uses the enhanced endpoint to insert documents with full metadata
   - Tracks source URLs, sitemap identifiers, and document indices
   - Provides better error handling and retry logic

## How It Works

### Document ID Generation
- Document IDs are generated using MD5 hash of the content (stripped)
- Format: `doc-{md5_hash}`
- This matches LightRAG's internal ID generation

### Metadata Enrichment
When inserting documents, the enhanced API:
1. Prepends metadata headers to the content:
   ```
   [SOURCE_URL: https://example.com/page]
   [SITEMAP: https://example.com/sitemap.xml]
   [INDEXED: 2024-01-01T12:00:00]
   [DOC_INDEX: 1 of 50]
   ```
2. Stores metadata in a persistent store
3. Ensures file_path is always set

### API Endpoints

#### POST `/documents/text/enhanced`
Insert text with full metadata support:
```json
{
  "text": "Document content",
  "description": "Optional description",
  "source_url": "https://example.com/page",
  "sitemap_identifier": "[SITEMAP: https://example.com/sitemap.xml]",
  "doc_index": 1,
  "total_docs": 50
}
```

#### POST `/documents/text`
Standard text insertion (with file_path fix):
```json
{
  "text": "Document content",
  "description": "Optional description",
  "file_path": "custom/path.txt"  // Optional
}
```

#### GET `/documents`
Returns all documents with proper file_path handling (no more validation errors!)

#### GET `/documents/by-sitemap/{sitemap_identifier}`
Get all documents for a specific sitemap

#### DELETE `/documents/by-sitemap/{sitemap_identifier}`
Delete all documents for a specific sitemap

#### DELETE `/documents/by-id`
Delete specific documents by their IDs:
```json
{
  "doc_ids": ["doc-abc123", "doc-def456"]
}
```

## Deployment

1. Copy the files to your project directory:
   - `lightrag_extended_api.py`
   - `Dockerfile`
   - `docker-compose.yml`
   - `.env.example` (rename to `.env` and add your keys)

2. Build and run:
   ```bash
   docker-compose up -d
   ```

3. Import the enhanced n8n workflow and update the endpoint URLs to match your deployment

## Environment Variables

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-ada-002
WORKING_DIR=/app/data/rag_storage
HOST=0.0.0.0
PORT=9621
```

## n8n Workflow Configuration

The enhanced workflow:
1. Sets the sitemap URL and identifier
2. Deletes all existing documents for that sitemap
3. Fetches and parses the sitemap
4. Crawls each URL
5. Inserts documents with full metadata using the enhanced endpoint

Update these nodes with your deployment URLs:
- `Delete Old Sitemap Docs`: Update to your LightRAG instance URL
- `Insert to LightRAG Enhanced`: Update to your LightRAG instance URL

## Benefits

1. **No more validation errors** - All documents have proper file_path values
2. **Better document management** - Track sources, sitemaps, and metadata
3. **Bulk operations** - Delete all documents from a sitemap in one call
4. **Backward compatible** - Works with existing LightRAG features
5. **Persistent metadata** - Document metadata survives restarts

## Testing

1. Check health endpoint:
   ```bash
   curl http://localhost:9621/health
   ```

2. Insert a test document:
   ```bash
   curl -X POST http://localhost:9621/documents/text/enhanced \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Test document content",
       "source_url": "https://example.com/test",
       "sitemap_identifier": "[SITEMAP: test]"
     }'
   ```

3. View documents (no more errors!):
   ```bash
   curl http://localhost:9621/documents
   ```