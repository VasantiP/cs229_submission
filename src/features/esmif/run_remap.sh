#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Remapping chunk split ..."
python remap_chunks_split.py

echo "Done."
