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
2. **Push `chroma_db/` to Railway**
  Prereqs once: `brew install railway`, `railway login`, `railway link` (link the **backend** service).
3. **Reload the API on Railway**
  In the Railway dashboard: **Restart** the backend service.  
   Use a **full redeploy** only if you also **changed code** and pushed it to GitHub.

---

## Two Railway services (reminder)


| Service       | Role          | Typical public URL                |
| ------------- | ------------- | --------------------------------- |
| Backend (API) | FastAPI       | e.g. `…alma-ai…` / `…rejlersrag…` |
| Frontend      | React + nginx | e.g. `…alma-frontend…`            |


- **Frontend (Docker) service:** keep **`VITE_API_BASE_URL`** = backend base `https://…` (no path, no trailing slash). The **nginx** image substitutes it at **container start** into `nginx/templates/default.conf.template` and proxies `/api` to FastAPI. The **browser** only talks to your frontend origin (good for **mobile Safari**). Rebuild/redeploy the **frontend** image after changing it.
- **Backend service:** **FRONTEND_ORIGIN** = frontend `https://…` (no trailing slash) so the API still accepts direct calls (e.g. tools) from that origin.

### Mobile “Load failed” (Railway)

- Prefer this **nginx `/api` proxy** (same-origin from the phone). Do not rely on baking the API URL only into JS unless you know the build received `VITE_API_BASE_URL`.
- If you still call the API **cross-origin** from the browser, use **`https://…`** for the API URL (not `http`) and redeploy the frontend build that embeds it.
- **`FRONTEND_ORIGIN`** on the API should match the frontend URL you open.

---

## Persistent volume (recommended)

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

- `scripts/push_chroma_to_railway.sh` - uploads local `chroma_db/` to `/app/chroma_db` on the linked service.

