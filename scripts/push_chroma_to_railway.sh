#!/usr/bin/env bash
# Push local chroma_db/ to the linked Railway service (/app/chroma_db).
# Prereqs: brew install railway; railway login; railway link (backend service).
# After run: restart the API service on Railway so in-memory caches refresh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d chroma_db ]] || [[ -z "$(ls -A chroma_db 2>/dev/null)" ]]; then
  echo "chroma_db/ missing or empty. Run from repo root: python -m src.ingest"
  exit 1
fi

echo "Uploading chroma_db/ → Railway:/app/chroma_db (this may take a while) ..."
# Railway CLI argument parsing can be strict; keep commands simple and shell-free.
railway ssh mkdir -p /app/chroma_db
# Stream only contents of local chroma_db/ into mounted /app/chroma_db.
tar -cf - -C chroma_db . | railway ssh tar -xf - -C /app/chroma_db
echo "Done. Restart the backend service on Railway, then test the app."
