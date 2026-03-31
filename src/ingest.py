"""
src/ingest.py — Ingestion Engine
--------------------------------
Reads ALL PDFs from ./data using Datapizza + Docling, batch-embeds with OpenAI
text-embedding-3-small (batch_size=100 to avoid 429), and stores vectors in ChromaDB
with metadata: filename, page number.

Also indexes scraped Sweco StoryMaps text: ``data/sweco_storymaps_*_item_*/sections.md``
(same Chroma collection, metadata ``source_type=storymaps``) so RAG retrieval can use it
alongside PDFs.

Usage:
    python -m src.ingest
    or: from src.ingest import run_ingestion; run_ingestion()
    or: from src.ingest import ingest_storymaps_sections; ingest_storymaps_sections()

Place PDFs in ./data/ and set OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import os
import glob
import logging
import time
import uuid
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Import after dotenv so config sees env
from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DATA_DIR,
    CHUNK_SIZE,
)

from functools import lru_cache
from types import SimpleNamespace

# Batch size for OpenAI Embeddings API (avoids 429 Too Many Requests)
EMBED_BATCH_SIZE: int = 100
EMBED_MAX_RETRIES: int = 3
EMBED_RETRY_DELAY_SEC: float = 5.0
# Smaller batches reduce peak RAM / SQLite lock time during upsert (avoids macOS OOM "killed").
CHROMA_UPSERT_BATCH_SIZE: int = 10

# ─────────────────────────────────────────────────────────────────────────────
# Recursive, context-aware chunking (Swedish engineering layout)
# ─────────────────────────────────────────────────────────────────────────────
# Use the shared CHUNK_SIZE so ingestion behavior stays consistent with config.
RECURSIVE_CHUNK_SIZE: int = CHUNK_SIZE
# Keep overlap proportional to avoid context gaps at boundaries.
RECURSIVE_CHUNK_OVERLAP: int = int(CHUNK_SIZE * 0.2)
# Priority: keep paragraphs/lists intact before falling back mid-sentence.
RECURSIVE_SEPARATORS: list[str] = ["\n\n", "\n", ".", " ", ""]


@lru_cache(maxsize=1)
def _get_tiktoken_encoding():
    """Get a token encoder compatible with the embedding model."""
    try:
        import tiktoken
    except ImportError as e:
        raise RuntimeError("tiktoken not installed. Run: pip install tiktoken") from e

    try:
        return tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)
    except KeyError:
        # Fallback to a common encoding if the exact model mapping is missing.
        return tiktoken.get_encoding("cl100k_base")


def _tiktoken_length(text: str) -> int:
    """Length function for LangChain splitters (measured in tokens)."""
    enc = _get_tiktoken_encoding()
    return len(enc.encode(text or ""))


@lru_cache(maxsize=1)
def _get_recursive_text_splitter():
    """
    Recursive character splitter with explicit Swedish-document-friendly separators.

    Note: separator order matters; we try larger natural boundaries first.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            # Older LangChain locations
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Recursive chunking requires LangChain text splitters. "
                "Install: pip install langchain-text-splitters"
            ) from e

    return RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=RECURSIVE_SEPARATORS,
        # Measure length in tokens (important for consistent chunk sizing
        # across languages and document sizes).
        length_function=_tiktoken_length,
    )


def _split_datapizza_chunk_recursively(chunk) -> list:
    """
    Split a single parsed node/chunk into sub-chunks, preserving metadata.

    Ensures strict source tracking by copying original `.metadata` (including page info)
    to every produced sub-chunk.
    """
    splitter = _get_recursive_text_splitter()

    def _get_node_text(x) -> str:
        """Extract best-effort text from either dict or datapizza Node."""
        if isinstance(x, dict):
            # Some pipelines store the actual text under `text`, others under `content`.
            return str(x.get("text") or x.get("content") or "") or ""
        # DoclingParser uses Node.content / Node._content (not Node.text).
        for attr in ("text", "content", "_content"):
            v = getattr(x, attr, None)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    # Be tolerant to whether datapizza returns an object or a dict.
    if isinstance(chunk, dict):
        text = str(chunk.get("text") or chunk.get("content") or "") or ""
        meta = chunk.get("metadata") or {}
        base_id = chunk.get("id")
    else:
        text = _get_node_text(chunk)
        meta = getattr(chunk, "metadata", None) or {}
        base_id = getattr(chunk, "id", None)

    if not text or not isinstance(text, str):
        return []

    splits = splitter.split_text(text)
    if not splits:
        return []

    out = []
    for i, s in enumerate(splits):
        if not s or not str(s).strip():
            continue
        # Copy metadata so downstream modifications can't leak across chunks.
        new_meta = dict(meta)
        new_id = f"{base_id}-{i}" if base_id is not None else None
        out.append(SimpleNamespace(text=str(s), metadata=new_meta, id=new_id))
    return out


def collect_storymaps_sections_chunks(
    data_dir: str | None = None,
) -> tuple[list[SimpleNamespace], list[str]]:
    """
    Read scraped StoryMaps ``sections.md`` under ``data/sweco_storymaps_*_item_*/``,
    split with the same recursive splitter as PDFs, return (chunks, filenames) for embedding.

    Each chunk gets metadata: filename (e.g. ``..._item_7/sections.md``), page (1-based
    part index), source_type=storymaps. Stable ids: ``storymaps_<item_dir>_p<n>`` for upsert.
    """
    root = Path(data_dir or DATA_DIR)
    if not root.is_dir():
        return [], []

    chunks_out: list[SimpleNamespace] = []
    filenames_out: list[str] = []

    for item_dir in sorted(root.glob("sweco_storymaps_*_item_*")):
        if not item_dir.is_dir():
            continue
        md_path = item_dir / "sections.md"
        if not md_path.is_file():
            continue
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            log.warning("Skip %s: %s", md_path, e)
            continue
        if not text:
            continue

        display_name = f"{item_dir.name}/sections.md"
        splitter = _get_recursive_text_splitter()
        splits = splitter.split_text(text)
        part_idx = 0
        for s in splits:
            s = (s or "").strip()
            if not s:
                continue
            part_idx += 1
            meta = {
                "filename": display_name,
                "page": part_idx,
                "source_type": "storymaps",
            }
            cid = f"storymaps_{item_dir.name}_p{part_idx}"
            chunks_out.append(SimpleNamespace(text=s, metadata=dict(meta), id=cid))
            filenames_out.append(display_name)

    if chunks_out:
        log.info(
            "Prepared %d chunks from StoryMaps sections.md under %s",
            len(chunks_out),
            root,
        )
    return chunks_out, filenames_out


def ingest_storymaps_sections(data_dir: str | None = None) -> int:
    """
    Embed StoryMaps ``sections.md`` chunks into the same Chroma collection as PDFs.
    Uses stable ids (upsert). Returns number of chunks stored.

    Call after scraping, or when you add/change StoryMaps folders under ./data.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")

    chunks, fnames = collect_storymaps_sections_chunks(data_dir)
    if not chunks:
        log.warning("No StoryMaps sections.md found under %s", data_dir or DATA_DIR)
        return 0

    texts = [c.text for c in chunks]
    log.info("Embedding %d StoryMaps chunks...", len(texts))
    embeddings_list = _embed_texts_batched(texts, batch_size=EMBED_BATCH_SIZE)
    _write_to_chroma(chunks, embeddings_list, fnames)
    log.info("StoryMaps sections ingestion complete (%d chunks).", len(chunks))
    return len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# IngestionPipeline (parse + split only; we handle embedding and Chroma ourselves)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ingestion_pipeline():
    """Build Datapizza IngestionPipeline: DoclingParser only (we split recursively later)."""
    try:
        from datapizza.pipeline import IngestionPipeline
        from datapizza.modules.parsers.docling import DoclingParser
    except ImportError as e:
        raise RuntimeError(
            f"datapizza import failed: {e}. "
            "Run: pip install datapizza-ai datapizza-ai-parsers-docling"
        ) from e

    try:
        from datapizza.modules.parsers.docling.ocr_options import OCROptions, OCREngine
        _ocr = OCROptions(engine=OCREngine.NONE)
        try:
            # Disable table structure recognition (TableFormer ~1 GB) — not needed for text RAG.
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            _pipe_opts = PdfPipelineOptions()
            _pipe_opts.do_table_structure = False
            parser = DoclingParser(ocr_options=_ocr, pipeline_options=_pipe_opts)
        except (ImportError, TypeError):
            parser = DoclingParser(ocr_options=_ocr)
    except ImportError:
        parser = DoclingParser()

    return IngestionPipeline(
        modules=[
            parser,
        ],
        vector_store=None,
        collection_name=None,
    )


def _embed_texts_batched(
    texts: list[str],
    model: str = OPENAI_EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """
    Embed texts in batches via OpenAI API. Returns list of vectors in same order as texts.
    Uses batch_size to avoid 429 rate limits.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai not installed. Run: pip install openai") from e

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings: list[list[float]] = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(0, len(texts), batch_size)
    if tqdm:
        iterator = tqdm(iterator, total=num_batches, desc="Embedding batches", unit="batch")

    for start in iterator:
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_num = (start // batch_size) + 1
        if not tqdm:
            log.info(f"  Embedding batch {batch_num}/{num_batches} ({len(batch_texts)} chunks)...")

        for attempt in range(EMBED_MAX_RETRIES):
            try:
                response = client.embeddings.create(input=batch_texts, model=model)
                batch_embeddings = [None] * len(batch_texts)
                for item in response.data:
                    batch_embeddings[item.index] = item.embedding
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "timeout" in err_msg or "429" in err_msg or "rate" in err_msg:
                    if attempt < EMBED_MAX_RETRIES - 1:
                        log.warning(f"  Batch {batch_num} failed ({e}); retry in {EMBED_RETRY_DELAY_SEC}s...")
                        time.sleep(EMBED_RETRY_DELAY_SEC)
                    else:
                        log.error(f"  Batch {batch_num} failed after {EMBED_MAX_RETRIES} retries.")
                        raise
                else:
                    raise

    return all_embeddings


_MAX_META_STR_LEN = 300  # chars — ChromaDB stores metadata in SQLite; large values bloat the DB by GBs

def _chunk_metadata(chunk, filename: str) -> dict:
    """
    Build Chroma-safe metadata dict: only filename, page, and small scalar extras.

    Docling/datapizza chunks carry rich metadata (table structures, layout trees, bounding
    boxes) that can be 100+ KB per chunk when str()-ified.  Storing those in ChromaDB's
    SQLite tables causes multi-GB bloat for a few hundred chunks.  We keep only the fields
    that are actually used by the UI (filename, page, source_type) plus any other scalar
    value that is short enough to be safe.
    """
    meta = getattr(chunk, "metadata", None) or {}
    out = {"filename": filename}

    # Page — prefer explicit field, treat 0 as valid.
    page = None
    for key in ("page", "page_number", "page_no", "pagenumber", "pageNo"):
        if key in meta and meta[key] is not None:
            page = meta[key]
            break

    if page is None:
        out["page"] = 0
    else:
        if isinstance(page, (int, float)) and not isinstance(page, bool):
            out["page"] = int(page)
        else:
            s = str(page).strip()
            out["page"] = int(s) if s.isdigit() else 0

    # Copy only small, safe scalar extras (e.g. source_type).  Skip anything large or
    # complex — those are the Docling document-structure objects that cause DB bloat.
    for k, v in meta.items():
        if k in ("filename", "page", "page_number", "page_no"):
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str):
            if len(v) <= _MAX_META_STR_LEN:
                out[k] = v
            # else: silently skip — long strings are usually full-text content
        else:
            # Non-scalar (list, dict, object) — only keep if the string representation
            # is short enough to be safe in SQLite.
            s = str(v)
            if len(s) <= _MAX_META_STR_LEN:
                out[k] = s
            else:
                log.debug("Skipping large metadata field '%s' (%d chars) for chunk in %s", k, len(s), filename)
    return out


def _compact_chroma_log() -> None:
    """
    Manually compact ChromaDB's SQLite database (export → delete → reimport).

    Only needed for ChromaDB 1.x which had an append-only log that grew to 14+ GB.
    ChromaDB 0.6.x does NOT have this problem — do NOT call this automatically after
    every write, as it deletes the DB while old file handles are still open, causing
    macOS to keep the old files on disk alongside new ones (30+ GB of disk thrash).

    Call this standalone (e.g. via compact_db.py) only if the DB has grown unexpectedly large.
    """
    import shutil

    db_path = CHROMA_DB_PATH
    collection_name = COLLECTION_NAME

    try:
        import chromadb as _chromadb
    except ImportError:
        return

    log.info("Compacting ChromaDB write log (export → rebuild → reimport)...")
    try:
        # 1. Export live data through the existing singleton.
        _client, col = get_chroma_collection()
        result = col.get(include=["embeddings", "documents", "metadatas"], limit=1_000_000)
        ids        = result["ids"]
        embeddings = result["embeddings"]
        documents  = result["documents"]
        metadatas  = result["metadatas"]
        if not ids:
            log.info("Compaction skipped — collection is empty.")
            return

        # 2. Invalidate the singleton so the next call opens a fresh connection.
        get_chroma_collection.cache_clear()

        # 3. Nuke and rebuild the database directory.
        shutil.rmtree(db_path)
        os.makedirs(db_path)

        # 4. Reimport into the clean database.
        new_client = _chromadb.PersistentClient(path=db_path)
        new_col = new_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        batch = CHROMA_UPSERT_BATCH_SIZE
        for i in range(0, len(ids), batch):
            end = min(i + batch, len(ids))
            new_col.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        size_mb = sum(
            os.path.getsize(os.path.join(d, f))
            for d, _, fs in os.walk(db_path) for f in fs
        ) / 1024 / 1024
        log.info("Compaction done — %d chunks, DB is now %.1f MB.", len(ids), size_mb)

    except Exception as exc:
        log.warning("Compaction failed (non-fatal): %s", exc)


def _write_to_chroma(
    chunks: list,
    embeddings_list: list[list[float]],
    filenames: list[str],
) -> None:
    """Persist chunks and embeddings to ChromaDB with metadata (filename, page)."""
    _client, collection = get_chroma_collection()

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        # embeddings_list is parallel to chunks (same order as _embed_texts_batched input).
        # Use index i, not a separate counter — skipping empty chunks must not shift indices.
        if not getattr(chunk, "text", None):
            continue
        filename = filenames[i] if i < len(filenames) else "unknown.pdf"
        meta = _chunk_metadata(chunk, filename)
        raw_id = getattr(chunk, "id", None)
        ids.append(str(raw_id) if raw_id is not None else str(uuid.uuid4()))
        embeddings.append(embeddings_list[i])
        documents.append(chunk.text)
        metadatas.append(meta)

    if not ids:
        log.warning("No chunks to write to ChromaDB.")
        return

    log.info(
        "Upserting %d chunks into Chroma (path=%s, collection=%s), batch_size=%d...",
        len(ids),
        CHROMA_DB_PATH,
        COLLECTION_NAME,
        CHROMA_UPSERT_BATCH_SIZE,
    )
    n = len(ids)
    for start in range(0, n, CHROMA_UPSERT_BATCH_SIZE):
        end = min(start + CHROMA_UPSERT_BATCH_SIZE, n)
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        log.info("  Chroma upsert progress: %d / %d", end, n)
    log.info(f"Stored {len(ids)} chunks in ChromaDB collection '{COLLECTION_NAME}'.")


def run_ingestion() -> None:
    """
    Main ingestion: read all PDFs from ./data, parse with Docling, chunk, batch-embed,
    plus StoryMaps ``sections.md`` under ``sweco_storymaps_*_item_*``, then store in ChromaDB.
    """
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set. Use .env and set OPENAI_API_KEY.")
        raise SystemExit(1)

    if not os.path.isdir(DATA_DIR):
        log.error(f"Data directory '{DATA_DIR}' does not exist.")
        raise SystemExit(1)

    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True))
    if not pdf_files:
        log.warning(
            "No PDFs found in '%s' — only StoryMaps markdown (if any) will be indexed.",
            DATA_DIR,
        )

    pipeline = _build_ingestion_pipeline() if pdf_files else None
    all_chunks: list = []
    all_filenames: list[str] = []

    for pdf_path in pdf_files:
        name = os.path.basename(pdf_path)
        log.info(f"Parsing: {name}")
        try:
            assert pipeline is not None
            chunks = pipeline.run(file_path=pdf_path, metadata={"filename": name})
        except Exception as e:
            log.warning(f"  Skip {name}: {e}")
            continue
        if not chunks:
            continue
        # Recursively split using context-aware rules; preserve metadata per sub-chunk.
        for c in chunks:
            if not c:
                continue
            # Docling nodes usually store text in `content` / `_content` not `text`.
            if isinstance(c, dict):
                text = c.get("text") or c.get("content") or ""
            else:
                text = (
                    getattr(c, "text", None)
                    or getattr(c, "content", None)
                    or getattr(c, "_content", None)
                    or ""
                )
            if not text or not isinstance(text, str):
                continue
            sub_chunks = _split_datapizza_chunk_recursively(c)
            for sc in sub_chunks:
                all_chunks.append(sc)
                all_filenames.append(name)

    sm_chunks, sm_names = collect_storymaps_sections_chunks()
    all_chunks.extend(sm_chunks)
    all_filenames.extend(sm_names)

    if not all_chunks:
        log.error(
            "No chunks to index: no PDFs produced chunks and no StoryMaps sections.md found."
        )
        raise SystemExit(1)

    texts = [c.text for c in all_chunks]
    log.info(f"Batch-embedding {len(texts)} chunks (batch_size={EMBED_BATCH_SIZE})...")
    embeddings_list = _embed_texts_batched(texts, batch_size=EMBED_BATCH_SIZE)

    # Free Docling's PyTorch models before opening ChromaDB — both compete for RAM
    # and the combined footprint causes OOM on machines with ≤16 GB unified memory.
    # DOCLING_DEVICE=cpu so torch.mps.empty_cache() is a no-op here; plain gc is enough.
    # The sleep gives macOS time to actually reclaim the released pages before ChromaDB
    # opens its own memory-mapped files.
    import gc, time
    del pipeline
    gc.collect()
    time.sleep(2)
    log.info("Docling memory released. Writing to ChromaDB...")

    _write_to_chroma(all_chunks, embeddings_list, all_filenames)
    log.info("Ingestion complete. ChromaDB at: %s", CHROMA_DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers for the web UI (upload/remove/list)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_chroma_collection():
    """
    Return (client, collection) — singleton for the entire process lifetime.

    lru_cache ensures only ONE PersistentClient (and its Rust HNSW engine) is ever
    open at a time, regardless of how many times the sidebar, retriever, or ingest
    helpers call this function. Sharing one connection avoids the multi-GB memory
    overhead of duplicate ChromaDB Rust engine instances.
    """
    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError("chromadb not installed. Run: pip install chromadb") from e

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def list_indexed_filenames(limit: int = 200_000) -> list[str]:
    """
    List unique filenames currently present in the Chroma collection.
    For PoC-scale collections this scans metadata and deduplicates.
    """
    _client, collection = get_chroma_collection()
    items = collection.get(include=["metadatas"], limit=limit)
    metas = items.get("metadatas") or []
    names = sorted({(m or {}).get("filename", "okänd") for m in metas})
    return [n for n in names if n]


def delete_by_filename(filename: str) -> int:
    """
    Delete all chunks belonging to a given source filename.
    Returns the number of ids deleted (best-effort; depends on Chroma version).
    """
    _client, collection = get_chroma_collection()
    # Some chroma versions don't return deletion count; we estimate via pre-scan.
    before = collection.count()
    collection.delete(where={"filename": filename})
    after = collection.count()
    return max(before - after, 0)


def clear_collection() -> int:
    """Delete all items in the collection. Returns deleted count (best-effort)."""
    _client, collection = get_chroma_collection()
    before = collection.count()
    # Chroma requires ids or where; use ids in chunks to clear all.
    ids = collection.get(include=[], limit=1_000_000).get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return before


def ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> int:
    """
    Ingest a single uploaded PDF (bytes): parse → chunk → batch embed → write to Chroma.
    Returns number of chunks stored.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    if not pdf_bytes:
        return 0

    # DoclingParser expects a file path; write to a temp file.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        pipeline = _build_ingestion_pipeline()
        chunks = pipeline.run(file_path=tmp_path, metadata={"filename": filename})
        chunk_list = []
        for c in (chunks or []):
            text = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
            if not text:
                continue
            chunk_list.extend(_split_datapizza_chunk_recursively(c))
        if not chunk_list:
            return 0

        texts = [c.text for c in chunk_list]
        embeddings_list = _embed_texts_batched(texts, batch_size=EMBED_BATCH_SIZE)
        _write_to_chroma(chunk_list, embeddings_list, [filename] * len(chunk_list))
        return len(chunk_list)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    run_ingestion()
