# Enhanced LightRAG with Extended API and WebUI

This repository extends the LightRAG REST API to solve the file_path validation error, add enhanced document management capabilities, and restore the WebUI functionality.

## Problem Solved

The original LightRAG `/documents/text` endpoint doesn't set a `file_path` field when inserting documents, which causes a Pydantic validation error when retrieving documents via the `/documents` endpoint. This enhanced version fixes this issue, adds additional functionality, and maintains the WebUI.

## Key Features

1. **Extended REST API** (`lightrag_extended_api.py`):
   - Fixes the file_path validation error by ensuring all documents have a file_path
   - Adds `/documents/text/enhanced` endpoint for rich metadata insertion
   - Adds `/documents/by-sitemap/{sitemap_url}` for sitemap-specific queries
   - Adds `/documents/by-sitemap/{sitemap_url}` DELETE endpoint for bulk deletion (returns success even if no documents exist)
   - Maintains an in-memory metadata store that persists to disk
   - Fully compatible with existing LightRAG Python API
   - **Serves the LightRAG WebUI at `/webui`**

2. **WebUI Support**:
   - Automatically serves the LightRAG web interface at `http://localhost:9621/webui`
   - Root URL (`/`) redirects to the WebUI
   - Includes knowledge graph visualization
   - Full compatibility with the original LightRAG UI

3. **Enhanced n8n Workflow**:
   - Automatically attempts to delete old documents from a sitemap before re-indexing (handles 404 gracefully)
   - Uses the enhanced endpoint to insert documents with full metadata
   - Tracks source URLs and document indices
   - Provides better error handling and retry logic
   - Simplified: uses sitemap URL directly without redundant identifiers

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
  "sitemap_url": "https://example.com/sitemap.xml",
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

#### GET `/documents/by-sitemap/{sitemap_url}`
Get all documents for a specific sitemap URL (URL-encoded)

#### DELETE `/documents/by-sitemap/{sitemap_url}`
Delete all documents for a specific sitemap URL (URL-encoded). Returns success even if no documents exist.

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
1. Sets the sitemap URL
2. Attempts to delete all existing documents for that sitemap (continues on 404 if none exist)
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
6. **Simplified workflow** - No need for redundant sitemap identifiers
7. **Graceful error handling** - Deletion works even if no documents exist
8. **WebUI included** - Access the full LightRAG web interface with knowledge graph visualization

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
       "sitemap_url": "https://example.com/sitemap.xml"
     }'
   ```

3. View documents (no more errors!):
   ```bash
   curl http://localhost:9621/documents
   ```

## Accessing the WebUI

Once the server is running, you can access the LightRAG web interface:

1. Open your browser and navigate to: `http://localhost:9621/webui`
2. Or simply go to `http://localhost:9621` (automatically redirects to WebUI)

The WebUI provides:
- Document management interface
- Knowledge graph visualization
- Query interface with different modes
- System status and statistics

If the WebUI is not available, you can:
1. Check the logs for WebUI mounting messages
2. Build it manually from the LightRAG repository
3. Place the built files in the `webui` directory

### Building WebUI Manually (Optional)

If the WebUI wasn't automatically set up during the build, you can build it manually:

```bash
# Option 1: Run inside the container
docker exec -it <container_name> /app/build_webui.sh

# Option 2: Build locally and mount
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG/lightrag_webui
bun install --frozen-lockfile
bun run build --emptyOutDir
# Copy the built files from lightrag/api/webui to your local webui directory
```

## Migration from Previous Version

If you have existing data with the old `sitemap_identifier` format, use the migration script:

```bash
# Run inside the container
docker exec -it <container_name> python /app/migrate_metadata.py

# Or with a custom path
docker exec -it <container_name> python /app/migrate_metadata.py /path/to/document_metadata.json
```

This script will:
- Convert `sitemap_identifier` to `sitemap_url`
- Create a backup of your metadata
- Update the content summaries to use the new format