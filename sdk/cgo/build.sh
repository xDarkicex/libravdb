#!/bin/bash
set -e

echo "Building LibraVDB C-Shared Library..."

# Determine the extension based on OS
OS="$(uname -s)"
EXT="so"
if [ "$OS" = "Darwin" ]; then
    EXT="dylib"
elif [ "$OS" = "MINGW32_NT" ] || [ "$OS" = "MINGW64_NT" ]; then
    EXT="dll"
fi

OUTPUT="libravdb.$EXT"

# Build the shared library
go build -buildmode=c-shared -o "$OUTPUT" .

echo "Successfully built $OUTPUT"
