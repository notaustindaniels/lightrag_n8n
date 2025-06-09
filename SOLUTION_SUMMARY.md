# Solution Summary: Custom Document IDs for LightRAG WebUI

## Problem
Documents in the LightRAG WebUI were displaying with hash-based IDs (e.g., `doc-abc123...`) instead of the formatted IDs provided by n8n (e.g., `[docs.python.org] 3/library/functions`). The hover behavior showed `[unknown]` for these hash IDs.

## Root Cause
LightRAG was generating its own hash-based document IDs internally, ignoring the custom IDs we tried to provide. The WebUI displays whatever ID LightRAG stores internally in its `doc_status` storage.

## Solution
We implemented proper custom ID support by:

1. **Using LightRAG's `ids` parameter correctly**: When inserting documents, we now pass the n8n-provided `document_id` as a custom ID using the `ids` parameter in the `ainsert` method.

2. **Consistent metadata storage**: We store metadata using the same ID that LightRAG will use internally (either the custom ID or the hash ID).

3. **Smart document retrieval**: The `/documents` endpoint now properly handles both custom IDs and hash IDs, ensuring the WebUI always displays the correct formatted ID.

## Key Changes

### 1. Enhanced Text Insertion (lines 280-360)
```python
# When n8n provides a document_id, use it as both the doc_id and custom_id
if request.document_id:
    doc_id = request.document_id
    custom_id = request.document_id
    metadata_key = custom_id
else:
    # Compute hash ID for documents without custom IDs
    doc_id = compute_doc_id(enriched_content)
    custom_id = doc_id
    metadata_key = doc_id

# Store metadata with the correct key
metadata_store[metadata_key] = metadata_entry

# Insert with custom ID
await rag_instance.ainsert(enriched_content, ids=[custom_id], file_paths=[file_path])
```

### 2. Document Retrieval (lines 436-479)
```python
# The doc_id from LightRAG could be either our custom ID or a hash ID
for doc_id, doc_data in all_docs.items():
    # Try to get metadata using the doc_id first
    metadata = metadata_store.get(doc_id, {})
    
    # Handle custom IDs that might not have metadata
    if not metadata and doc_id.startswith('[') and ']' in doc_id:
        metadata = {"file_path": doc_id, "display_name": doc_id}
    
    # For display, use custom ID if available
    if doc_id.startswith('[') and ']' in doc_id:
        display_id = doc_id  # This is already the custom ID
    else:
        display_id = metadata.get('id', doc_id)  # Use metadata ID or fall back to hash
```

## Results
- Documents inserted with n8n's `document_id` now display with the correct format: `[domain] path`
- Hover behavior shows the full path correctly
- No more `[unknown]` prefixes in the WebUI
- Backward compatibility maintained for existing documents

## Testing
Use `test_custom_ids.py` to verify the implementation:
```bash
python test_custom_ids.py
```

This will test:
1. Inserting documents with custom IDs
2. Retrieving documents and verifying correct display
3. Simulating hover behavior