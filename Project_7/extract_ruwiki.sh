#!/usr/bin/env bash
set -euo pipefail

# Paths (adjust if needed)
PROJECT_ROOT="$HOME/Project_7"
DUMP_DIR="$PROJECT_ROOT/ruwiki"
DUMP_BZ2="$DUMP_DIR/ruwiki-latest-pages-articles.xml.bz2"
DUMP_XML="$DUMP_DIR/ruwiki-latest-pages-articles.xml"
OUTPUT_DIR="$DUMP_DIR/extracted"

echo "== Russian Wikipedia extraction =="
# 2. Decompress (if not already decompressed)
if [[ ! -f "$DUMP_XML" ]]; then
  echo "Decompressing $DUMP_BZ2 ..."
  bzip2 -dk "$DUMP_BZ2"
else
  echo "Decompressed XML already exists at: $DUMP_XML"
fi

# 3. Create output directory
mkdir -p "$OUTPUT_DIR"

# 4. Check if extraction already exists
#    If there are files named wiki_* inside OUTPUT_DIR, assume we've already extracted.
if find "$OUTPUT_DIR" -name 'wiki_*' -type f -print -quit | grep -q .; then
  echo
  echo "✓ Extraction output already found in: $OUTPUT_DIR"
  echo "  (Skipping WikiExtractor to avoid re-running the long extraction step.)"
else
  echo
  echo "Running WikiExtractor..."
  # NOTE: no --min-text-length here; your wikiextractor doesn't support it.
  wikiextractor \
    -o "$OUTPUT_DIR" \
    --processes "$(nproc)" \
    --no-templates \
    "$DUMP_XML"

  echo
  echo "✓ Extraction complete!"
fi

echo
echo "Done."
echo "You can inspect some output, for example:"
echo "  head -n 40 $(find \"$OUTPUT_DIR\" -name 'wiki_*' | head -n 1)"

