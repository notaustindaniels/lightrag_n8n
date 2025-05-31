#!/bin/bash
# Script to download pre-built LightRAG WebUI files

echo "Setting up LightRAG WebUI..."

# Create webui directory
mkdir -p /app/webui

# Function to check if webui has files
check_webui() {
    if [ -d "$1" ] && [ "$(ls -A $1)" ]; then
        return 0
    else
        return 1
    fi
}

# Try to copy from installed package first
echo "Looking for WebUI in installed package..."
for path in "/usr/local/lib/python3.11/site-packages/lightrag/api/webui" \
            "/usr/local/lib/python3.10/site-packages/lightrag/api/webui" \
            "/usr/local/lib/python3.9/site-packages/lightrag/api/webui"; do
    if check_webui "$path"; then
        echo "Copying WebUI from $path..."
        cp -r "$path"/* /app/webui/
        echo "WebUI copied successfully!"
        exit 0
    fi
done

# Try Python import method
echo "Trying to locate WebUI via Python import..."
LIGHTRAG_PATH=$(python -c "import lightrag; import os; print(os.path.dirname(lightrag.__file__))" 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$LIGHTRAG_PATH" ]; then
    WEBUI_PATH="$LIGHTRAG_PATH/api/webui"
    if check_webui "$WEBUI_PATH"; then
        echo "Found WebUI at $WEBUI_PATH"
        cp -r "$WEBUI_PATH"/* /app/webui/
        echo "WebUI copied successfully!"
        exit 0
    fi
fi

# If not found, try to build from source
echo "WebUI not found in installed package. Attempting to build from source..."

# Clone LightRAG repository temporarily
TEMP_DIR="/tmp/lightrag_webui_build"
rm -rf "$TEMP_DIR"
git clone --depth 1 https://github.com/HKUDS/LightRAG.git "$TEMP_DIR" 2>/dev/null

if [ -d "$TEMP_DIR/lightrag_webui" ]; then
    echo "Found WebUI source. Checking for pre-built files..."
    
    # Check if there are pre-built files
    if check_webui "$TEMP_DIR/lightrag/api/webui"; then
        echo "Found pre-built WebUI files!"
        cp -r "$TEMP_DIR/lightrag/api/webui"/* /app/webui/
        echo "WebUI copied successfully!"
    else
        echo "No pre-built files found."
        
        # Create a minimal index.html as fallback
        cat > /app/webui/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>LightRAG - WebUI Not Available</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 5px; }
        code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LightRAG Extended API</h1>
        <div class="warning">
            <h2>WebUI Not Available</h2>
            <p>The LightRAG WebUI files were not found during the build process.</p>
            <p>You can still use the API endpoints directly:</p>
            <ul>
                <li>API Documentation: <a href="/docs">/docs</a></li>
                <li>Health Check: <a href="/health">/health</a></li>
            </ul>
            <p>To enable the WebUI, you can:</p>
            <ol>
                <li>Build the WebUI from the LightRAG repository</li>
                <li>Place the built files in the <code>webui</code> directory</li>
                <li>Restart the container</li>
            </ol>
        </div>
    </div>
</body>
</html>
EOF
        echo "Created fallback index.html"
    fi
else
    echo "Could not clone LightRAG repository."
    # Create minimal fallback
    echo "<html><body><h1>WebUI not available</h1><p>API is running. See <a href='/docs'>/docs</a></p></body></html>" > /app/webui/index.html
fi

# Cleanup
rm -rf "$TEMP_DIR"

# Set permissions
chmod -R 755 /app/webui

echo "WebUI setup complete!"