"""
compact_db.py — run this to shrink the ChromaDB SQLite from GB to MB.

ChromaDB 1.x uses an append-only write log that never compacts itself.
Every ingest/delete cycle appends to the log forever. This script exports
the live vectors, nukes the bloated file, and reimports cleanly.

Usage (with venv activated):
    python compact_db.py
"""
import chromadb, shutil, os, sys
from pathlib import Path

DB_PATH   = str(Path(__file__).parent / "chroma_db")
COLLECTION = "rejlers_documents"

def main():
    size_before = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, fs in os.walk(DB_PATH) for f in fs
    ) / 1024**3
    print(f"Database size before: {size_before:.2f} GB")

    print("Step 1 — exporting live chunks...")
    client = chromadb.PersistentClient(path=DB_PATH)
    col    = client.get_collection(COLLECTION)
    result = col.get(include=["embeddings", "documents", "metadatas"], limit=1_000_000)
    ids        = result["ids"]
    embeddings = result["embeddings"]
    documents  = result["documents"]
    metadatas  = result["metadatas"]
    print(f"  Exported {len(ids)} chunks.")
    del col, client

    print("Step 2 — deleting bloated database...")
    shutil.rmtree(DB_PATH)
    os.makedirs(DB_PATH)

    print("Step 3 — writing clean database...")
    client2 = chromadb.PersistentClient(path=DB_PATH)
    col2    = client2.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    BATCH = 10
    for i in range(0, len(ids), BATCH):
        end = min(i + BATCH, len(ids))
        col2.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"  {end}/{len(ids)}", end="\r")

    print(f"\n  Done. {col2.count()} chunks in fresh database.")
    size_after = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, fs in os.walk(DB_PATH) for f in fs
    ) / 1024**2
    print(f"Database size after:  {size_after:.1f} MB")

if __name__ == "__main__":
    main()
