#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# Download Quora Question Pairs (GLUE release) into data/qqp.
# Safe to re-run: skips files that are already there.
#
# Source: https://huggingface.co/datasets/nyu-mll/glue (config "qqp").
# The data is not redistributed by this repository; its use is subject to
# Quora's terms. Reading .parquet needs pyarrow: pip install -e ".[bench]"
# --------------------------------------------------------------------------- #
set -euo pipefail

DEST="${1:-data/qqp}"
BASE="https://huggingface.co/datasets/nyu-mll/glue/resolve/main/qqp"
SPLITS=("train" "validation")

log() { printf '[bench-cache-data] %s\n' "$*" >&2; }

if ! command -v curl >/dev/null 2>&1; then
    log "curl is required"
    exit 1
fi

mkdir -p "${DEST}"
for split in "${SPLITS[@]}"; do
    target="${DEST}/${split}.parquet"
    if [[ -s "${target}" ]]; then
        log "${target} already present"
        continue
    fi
    log "downloading ${split} split"
    # Write to a temp file first so an interrupted download never looks complete.
    curl -fL --retry 3 --retry-delay 2 -o "${target}.part" \
        "${BASE}/${split}-00000-of-00001.parquet"
    mv "${target}.part" "${target}"
done

if ! python -c "import pyarrow" 2>/dev/null; then
    log 'note: install pyarrow to read these files: pip install -e ".[bench]"'
fi
log "done: $(du -sh "${DEST}" | cut -f1) in ${DEST}"
