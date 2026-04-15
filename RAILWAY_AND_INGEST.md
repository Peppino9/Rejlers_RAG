# Railway + local ingestion (cheat sheet)

`chroma_db/` and `data/` are **not** in Git. The API on Railway does **not** get a new index from `git push` alone.

---

## Day-to-day: ship a new index after you change ingestion or documents

Run on **your Mac**, from the repo root (`Rejlers_RAG/`):

1. **Ingest (rebuild local index)**
  ```bash
   python -m src.ingest
  ```
   Optional clean slate first:  
   `rm -rf chroma_db && mkdir -p chroma_db` then run ingest again.
2. **Create archive**
  ```bash
   tar -cf /tmp/chroma_db.tar -C chroma_db .
   ls -lh /tmp/chroma_db.tar
  ```
3. **Upload archive to Google Drive (or other public file host)**
  - Set sharing to **Anyone with the link**.
  - Keep the file id (example: `1jCxJIA-9H1jiw1Hzh1MxBQtofoCp6_di`).
4. **Open Railway shell to backend and replace `/app/chroma_db`**
  ```bash
   railway ssh --project=<PROJECT_ID> --environment=<ENV_ID> --service=<BACKEND_SERVICE_ID>
   mkdir -p /app/chroma_db
   find /app/chroma_db -mindepth 1 -delete
   python -m pip install --no-cache-dir gdown
   python -m gdown "https://drive.google.com/uc?id=<FILE_ID>" -O /tmp/chroma_db.tar
   ls -lh /tmp/chroma_db.tar
   tar -xf /tmp/chroma_db.tar -C /app/chroma_db
   ls -lah /app/chroma_db
   exit
  ```
5. **Reload the API on Railway**
  - In the Railway dashboard: **Restart** the backend service.
  - Use a **full redeploy** only if you also changed code.

---

## Two Railway services (reminder)


| Service       | Role          | Typical public URL                |
| ------------- | ------------- | --------------------------------- |
| Backend (API) | FastAPI       | e.g. `…alma-ai…` / `…rejlersrag…` |
| Frontend      | React + nginx | e.g. `…alma-frontend…`            |


- Frontend env: `**VITE_API_BASE_URL`** = backend `https://…` (exact name, rebuild frontend after changes).  
- Backend env: `**FRONTEND_ORIGIN`** = frontend `https://…` (no trailing slash).

---

## Persistent volume (recommended, strongly)

Mount a Railway **volume** on `**/app/chroma_db`** so the index survives redeploys.  
If you only push to the container filesystem with **no** volume, a new deploy can wipe the DB unless you push again.

---

## How to know you’re on the **newest** index

- Ingest and push use the `**config.py` / `src/ingest.py`** version on your machine **at ingest time**. Pull the branch you want before ingesting.
- After push, compare **folder age / size**:
  - Mac: `ls -la chroma_db`
  - Railway (`railway ssh`): `ls -la /app/chroma_db`
- Optional **chunk count** (should match after a successful push):
**Mac (repo root):**
  ```bash
  python -c "from src.ingest import get_chroma_collection; _, c = get_chroma_collection(); print('chunks:', c.count())"
  ```
  **Railway (`railway ssh`, then):**
  ```bash
  cd /app && python -c "from src.ingest import get_chroma_collection; _, c = get_chroma_collection(); print('chunks:', c.count())"
  ```

---

## Health check

- Backend: `GET https://<your-api-host>/health` → `{"status":"ok"}`  
- Root `/` may return a small JSON pointer; **ingestion is unrelated** to `/health`.

---

## Railway CLI shell (ingest on server only if you want)

From Mac, repo linked to backend:

```bash
railway ssh
# then e.g. cd /app && python -m src.ingest
```

That needs `**/app/data**` with PDFs on the server. For normal iteration, **ingest on Mac + push script** is simpler.

---

## Script location

- `scripts/push_chroma_to_railway.sh` — helper script for direct CLI upload.
- If Railway SSH stdin/tar piping hangs or fails, use the Google Drive + `gdown` flow above (this is the most reliable fallback).

