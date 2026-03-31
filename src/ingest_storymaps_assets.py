"""
ingest_storymaps_assets.py — Index StoryMaps scraped assets into Chroma
---------------------------------------------------------------------------
This PoC RAG pipeline currently ingests only PDFs.

Your StoryMaps scraper downloads:
  - images: data/sweco_storymaps_*_item_*/images/*.jpg|png
  - webmaps: data/sweco_storymaps_*_item_*/webmaps/*.json
  - manifest: assets_manifest.json and meta.json in each item folder

To make images/maps retrievable + renderable in the UI:
  - we embed alt/caption + map layer titles/URLs into Chroma documents
  - we store metadata like asset_path + asset_type

Usage:
  python -m src.ingest_storymaps_assets
  python -m src.ingest_storymaps_assets --items 3 4 5 6 7
  python -m src.ingest_storymaps_assets --skip-webmaps
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from config import CHROMA_DB_PATH, COLLECTION_NAME, OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY


DATA_DIR = Path("./data")


@dataclass(frozen=True)
class AssetChunk:
    id: str
    text: str
    metadata: dict


def _clean_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_webmap_layer_titles(webmap_json: dict) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()

    for ol in webmap_json.get("operationalLayers") or []:
        if not isinstance(ol, dict):
            continue
        layers = ol.get("layers")
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            t = layer.get("title")
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t or t in seen:
                continue
            titles.append(t)
            seen.add(t)

    # Stable, human-first order.
    return titles


def _embed_texts_batched(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai not installed. Run: pip install openai") from e

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        resp = client.embeddings.create(input=batch, model=OPENAI_EMBEDDING_MODEL)
        batch_embeddings = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _iter_storymaps_item_dirs() -> Iterable[Path]:
    # Expected folder naming: sweco_storymaps_<collectionId>_item_<n>
    yield from sorted(Path(DATA_DIR).glob("sweco_storymaps_*_item_*"))


def _parse_item_number(item_dir: Path) -> int | None:
    m = re.search(r"_item_(\d+)$", item_dir.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _build_asset_chunks_for_item(item_dir: Path, skip_webmaps: bool) -> list[AssetChunk]:
    manifest_path = item_dir / "assets_manifest.json"
    meta_path = item_dir / "meta.json"

    if not manifest_path.exists():
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    story_id = meta.get("active_story_id") or ""
    item_number = _parse_item_number(item_dir)

    chunks: list[AssetChunk] = []

    repo_root = Path(__file__).resolve().parents[1]
    # Ensure absolute paths for Streamlit st.image rendering.
    abs_item_dir = (repo_root / item_dir).resolve()

    # Images
    for img in manifest.get("images") or []:
        if not isinstance(img, dict):
            continue
        node_id = img.get("node_id") or ""
        resource_file = img.get("resource_file")
        if not isinstance(resource_file, str) or not resource_file:
            continue
        alt = img.get("alt") or ""
        caption = img.get("caption") or ""

        asset_path = (abs_item_dir / "images" / resource_file).resolve()
        if not asset_path.exists():
            # Skip missing local downloads.
            continue

        # Short document that embedding can match to queries.
        doc = _clean_ws(
            "\n".join(
                [
                    "StoryMaps image",
                    f"item={item_dir.name}",
                    f"active_story_id={story_id}",
                    f"alt={alt}",
                    f"caption={caption}",
                ]
            )
        )

        chunk_id = f"storymaps_asset_image_{item_dir.name}_{resource_file}_{node_id}"
        metadata = {
            "filename": chunk_id,
            "page": 0,
            "asset_type": "image",
            "asset_path": str(asset_path),
            "item_dir": item_dir.name,
            "resource_file": resource_file,
            "node_id": str(node_id),
            "alt": alt[:500],
            "caption": caption[:500],
            "active_story_id": story_id,
            "item_number": item_number if item_number is not None else -1,
        }

        chunks.append(AssetChunk(id=chunk_id, text=doc, metadata=metadata))

    # Webmaps (optional)
    if not skip_webmaps:
        for wm in manifest.get("webmaps") or []:
            if not isinstance(wm, dict):
                continue
            node_id = wm.get("node_id") or ""
            webmap_item_id = wm.get("webmap_item_id")
            if not isinstance(webmap_item_id, str) or not webmap_item_id:
                continue

            webmap_json_path = (abs_item_dir / "webmaps" / f"{webmap_item_id}.json").resolve()
            if not webmap_json_path.exists():
                continue

            webmap_json = json.loads(webmap_json_path.read_text(encoding="utf-8"))

            caption = wm.get("caption") or ""
            layer_titles = _extract_webmap_layer_titles(webmap_json)
            urls = wm.get("visible_layers_urls") or []
            if not isinstance(urls, list):
                urls = []
            urls_str = ", ".join([u for u in urls[:25] if isinstance(u, str)])

            doc = _clean_ws(
                "\n".join(
                    [
                        "StoryMaps webmap",
                        f"item={item_dir.name}",
                        f"active_story_id={story_id}",
                        f"caption={caption}",
                        "layer_titles=" + (", ".join(layer_titles[:60]) if layer_titles else ""),
                        "visible_layer_urls=" + urls_str,
                    ]
                )
            )

            chunk_id = f"storymaps_asset_webmap_{item_dir.name}_{webmap_item_id}_{node_id}"
            metadata = {
                "filename": chunk_id,
                "page": 0,
                "asset_type": "webmap",
                "asset_path": str(webmap_json_path),
                "item_dir": item_dir.name,
                "webmap_item_id": webmap_item_id,
                "node_id": str(node_id),
                "caption": str(caption)[:500],
                "layer_titles": json.dumps(layer_titles[:80], ensure_ascii=False),
                "active_story_id": story_id,
                "item_number": item_number if item_number is not None else -1,
            }

            chunks.append(AssetChunk(id=chunk_id, text=doc, metadata=metadata))

    return chunks


def run_ingestion(items: list[int] | None, skip_webmaps: bool) -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing. Set it in .env.")

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError("chromadb not installed. Run: pip install chromadb") from e

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[AssetChunk] = []

    for item_dir in _iter_storymaps_item_dirs():
        item_number = _parse_item_number(item_dir)
        if item_number is None:
            continue
        if items is not None and item_number not in items:
            continue

        log.info(f"Indexing assets for: {item_dir.name}")
        chunks = _build_asset_chunks_for_item(item_dir=item_dir, skip_webmaps=skip_webmaps)
        log.info(f"  -> {len(chunks)} asset chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        log.warning("No asset chunks found to ingest.")
        return

    # Embed + upsert
    texts = [c.text for c in all_chunks]
    ids = [c.id for c in all_chunks]
    embeddings = _embed_texts_batched(texts, batch_size=100)
    metadatas = [c.metadata for c in all_chunks]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    log.info(
        "Done indexing StoryMaps assets. "
        f"Upserted: {len(all_chunks)} chunks into '{COLLECTION_NAME}'."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="*", type=int, help="StoryMaps item numbers to index (e.g. 3 4 5)")
    ap.add_argument("--skip-webmaps", action="store_true", help="Index only images (not webmap JSON).")
    args = ap.parse_args()

    run_ingestion(items=args.items if args.items else None, skip_webmaps=bool(args.skip_webmaps))


if __name__ == "__main__":
    main()

