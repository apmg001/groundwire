#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# Download the RAGTruth benchmark (MIT licence) into data/ragtruth.
# Safe to re-run: skips the download if the files are already there.
#
# Source: https://github.com/ParticleMedia/RAGTruth
# --------------------------------------------------------------------------- #
set -euo pipefail

DEST="${1:-data/ragtruth}"
REPO="https://github.com/ParticleMedia/RAGTruth.git"
FILES=("dataset/response.jsonl" "dataset/source_info.jsonl")

log() { printf '[bench-data] %s\n' "$*" >&2; }

all_present() {
    for f in "${FILES[@]}"; do [[ -s "${DEST}/${f}" ]] || return 1; done
}

if all_present; then
    log "RAGTruth already present in ${DEST}"
    exit 0
fi

if ! command -v git >/dev/null 2>&1; then
    log "git is required to download RAGTruth"
    exit 1
fi

if [[ -e "${DEST}" ]]; then
    log "${DEST} exists but is incomplete; remove it and re-run"
    exit 1
fi

log "cloning ${REPO} into ${DEST} (~36 MB)"
mkdir -p "$(dirname "${DEST}")"
git clone --depth 1 --quiet "${REPO}" "${DEST}"

if ! all_present; then
    log "download finished but expected files are missing: ${FILES[*]}"
    exit 1
fi
log "done: $(wc -l < "${DEST}/dataset/response.jsonl") responses"
