#!/bin/bash
# Manual script to build LightRAG WebUI from source
# Run this if the automatic download didn't work

echo "Building LightRAG WebUI from source..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is required but not installed. Installing..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y nodejs
fi

# Check if Bun is installed (preferred for LightRAG WebUI)
if ! command -v bun &> /dev/null; then
    echo "Installing Bun..."
    curl -fsSL https://bun.sh/install | bash
    export PATH="$HOME/.bun/bin:$PATH"
fi

# Clone LightRAG repository
TEMP_DIR="/tmp/lightrag_build"
rm -rf "$TEMP_DIR"
echo "Cloning LightRAG repository..."
git clone https://github.com/HKUDS/LightRAG.git "$TEMP_DIR"

# Navigate to WebUI directory
cd "$TEMP_DIR/lightrag_webui" || {
    echo "Error: lightrag_webui directory not found"
    exit 1
}

# Install dependencies and build
echo "Installing dependencies..."
if command -v bun &> /dev/null; then
    bun install --frozen-lockfile
    echo "Building WebUI with Bun..."
    bun run build --emptyOutDir
else
    npm install
    echo "Building WebUI with npm..."
    npm run build
fi

# Copy built files
if [ -d "$TEMP_DIR/lightrag/api/webui" ]; then
    echo "Copying built WebUI files..."
    mkdir -p /app/webui
    cp -r "$TEMP_DIR/lightrag/api/webui"/* /app/webui/
    echo "WebUI built and installed successfully!"
else
    echo "Error: Built files not found at expected location"
    exit 1
fi

# Cleanup
cd /
rm -rf "$TEMP_DIR"

echo "Build complete! Restart the container to use the new WebUI."