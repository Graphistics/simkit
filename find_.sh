#!/bin/bash

# Search from the root directory for .java files containing the target string
# Requires sudo to access all system files

SEARCH_TERM="spectralClustering_validParams_featureBased_fullGraph"
SEARCH_PATH="/"
FILE_EXTENSION=".java"

echo "Searching for '$SEARCH_TERM' inside $FILE_EXTENSION files from root ($SEARCH_PATH)..."
echo "This may take a while and requires sudo permission."

sudo find "$SEARCH_PATH" -type f -name "*$FILE_EXTENSION" 2>/dev/null | while read -r file; do
    if grep -q "$SEARCH_TERM" "$file"; then
        echo "Found in: $file"
    fi
done