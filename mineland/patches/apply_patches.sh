#!/bin/bash
# Apply prismarine-viewer-colalab patches for overhead camera support
# Run after: cd mineland/sim/mineflayer && npm ci  (this script lives in mineland/patches/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/../sim/mineflayer/node_modules/prismarine-viewer-colalab/lib"

if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Target directory not found: $TARGET_DIR"
    echo "Run 'npm ci' in mineland/sim/mineflayer/ first."
    exit 1
fi

cp "$SCRIPT_DIR/freecamera-base64.js" "$TARGET_DIR/freecamera-base64.js"
cp "$SCRIPT_DIR/headless-base64.js" "$TARGET_DIR/headless-base64.js"

VIEWER_DIR="$SCRIPT_DIR/../sim/mineflayer/node_modules/prismarine-viewer-colalab/viewer/lib"
cp "$SCRIPT_DIR/Entity.js" "$VIEWER_DIR/entity/Entity.js"
cp "$SCRIPT_DIR/entities.js" "$VIEWER_DIR/entities.js"
cp "$SCRIPT_DIR/worldView.js" "$VIEWER_DIR/worldView.js"

echo "Patches applied successfully."
