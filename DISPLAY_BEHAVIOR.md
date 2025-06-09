# Document Display Behavior in LightRAG WebUI

## Current Display Fields

The WebUI uses these fields from the API response:

1. **ID Display**: Shows the `source_url` from metadata
   - This displays the full URL of the original document
   - Example: `https://docs.python.org/3/library/functions/decorators/advanced-usage`
   - This is the "cool feature" that shows where the document came from!

2. **File Name Display**: Shows the `display_name` field
   - Should show: `[domain] slug`
   - Example: `[docs.python.org] advanced-usage`
   - This is what users see in the list

3. **Hover Display**: Shows the `file_path` field
   - Should show: `[domain] full_path`
   - Example: `[docs.python.org] 3/library/functions/decorators/advanced-usage`
   - This provides full context on hover

## How It Works

### When n8n provides a document_id:
```
document_id: "[docs.python.org] 3/library/functions/decorators/advanced-usage"
```

The API:
1. Uses this as the `file_path` (for hover display)
2. Extracts the slug to create `display_name`: `[docs.python.org] advanced-usage`
3. Stores the `source_url` in metadata (for ID display)

### Result in WebUI:
- **ID**: `https://docs.python.org/3/library/functions/decorators/advanced-usage` (full URL)
- **File Name**: `[docs.python.org] advanced-usage` (domain + slug)
- **Hover**: `[docs.python.org] 3/library/functions/decorators/advanced-usage` (domain + full path)

## The "Cool" ID Feature

The ID showing the full URL is actually a side effect of the WebUI displaying the `source_url` field from metadata. This is unexpectedly useful because:
- Users can see the exact source of each document
- It's clickable and can be copied
- Provides full transparency about document origins

This behavior is preserved in the current implementation!