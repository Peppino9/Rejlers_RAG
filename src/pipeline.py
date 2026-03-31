"""
src/pipeline.py — RAG DAG
-------------------------
Datapizza DagPipeline: embed query → retrieve top-k from ChromaDB → generate with
OpenAI gpt-4o-mini. Two toggleable modes for A/B testing:
  - Expert (Prompt A): technical, exact, professional.
  - Citizen (Prompt B): simplified language, short sentences, low LIX; no omission of facts.

Output always includes strict Source Tracking at the end:
  "Källor: fil.pdf (sid 12), bilaga.pdf (sid 4)"
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    LLM_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K,
    SYSTEM_PROMPTS,
    RAG_USER_INSTRUCTIONS,
    PROMPT_MAX_CHUNKS,
    PROMPT_MAX_CONTEXT_CHARS,
    PROMPT_MAX_CHUNK_CHARS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query retrieval: variants improve recall when questions use "Hur …?" etc.
# ─────────────────────────────────────────────────────────────────────────────

_RETRIEVAL_STOP: frozenset[str] = frozenset(
    {
        "och", "att", "som", "för", "med", "till", "inte", "kan", "har", "den", "det",
        "detta", "dessa", "är", "var", "blir", "från", "vid", "per", "inom",
    }
)


def _retrieval_query_variants(query: str) -> list[str]:
    """Build 1–3 search strings: full question, stripped question words, keyword bundle."""
    q = (query or "").strip()
    if not q:
        return [""]
    variants: list[str] = [q]
    stripped = re.sub(
        r"^\s*(hur|vad|varför|när|vilken|vilka|är|finns|kan|skulle|borde|måste|ska)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if stripped and stripped.lower() != q.lower() and len(stripped) > 3:
        variants.append(stripped)
    words = re.findall(r"[A-Za-zÅÄÖåäö0-9\-]{4,}", q.lower())
    kws = [w for w in words if w not in _RETRIEVAL_STOP]
    if kws:
        variants.append(" ".join(kws[:10]))
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)

    # Järnväg: frågor om "bank/banker" matchar ofta PDF-formuleringar som "stråk", "på bank",
    # "orange stråk", "linjeföring" — extra söksträng ökar recall.
    ql = q.lower()
    if re.search(r"\bbank|banker\b", ql):
        syn = f"{q} stråk korridor linjeföring bro tunnel bansträckning järnvägsplan"
        sl = syn.lower().strip()
        if sl and sl not in seen:
            seen.add(sl)
            out.append(syn.strip())
        # Ren nyckelordsrad: samrådshandlingar beskriver ofta "orange stråk", "på bank" utan ordet "banker".
        rail_kw = (
            "orange stråk grön stråk linjeföring järnväg bank bro tunnel korridor "
            "bansträckning utformning samråd"
        )
        if rail_kw.lower() not in seen:
            seen.add(rail_kw.lower())
            out.append(rail_kw)

    # Järnvägsfrågor utan ordet "bank": komplettera med stråk-termer.
    if re.search(r"järnväg|järnvägs|bansträckning|korridor", ql) and not re.search(
        r"\bbank|banker\b", ql
    ):
        rail_kw2 = (
            "järnväg stråk orange grön linjeföring bank bro tunnel samråd utformning"
        )
        if rail_kw2.lower() not in seen:
            seen.add(rail_kw2.lower())
            out.append(rail_kw2)

    # Fastighets-/närboendefrågor: underlag använder ofta markåtkomst, buller, nyttjande, MKB.
    if re.search(
        r"fastighet|fastigheter|närboend|grann|markäg|tomt|bostad|bebygg", ql
    ) and re.search(
        r"järnväg|järnvägs|linje|korridor|stråk|bansträck|ny.*linj", ql
    ):
        prop_kw = (
            "fastighet markåtkomst markintrång nyttjande ersättning buller vibration bostad "
            "omgivningspåverkan trädsäkringszon skyddsåtgärd bullerskydd närboende samråd MKB"
        )
        if prop_kw.lower() not in seen:
            seen.add(prop_kw.lower())
            out.append(prop_kw)
        prop_q = f"{q} omgivningspåverkan mark buller nyttjande skyddsåtgärder"
        pq = prop_q.lower().strip()
        if pq not in seen:
            seen.add(pq)
            out.append(prop_q.strip())

    return out if out else [q]


def _chunk_dedupe_key(text: str | None, meta: dict) -> tuple[str, str, int]:
    fn = str((meta or {}).get("filename") or "")
    page = (meta or {}).get("page", 0)
    try:
        p = int(page)
    except (TypeError, ValueError):
        p = 0
    head = (text or "")[:240]
    return (head, fn, p)


# ─────────────────────────────────────────────────────────────────────────────
# Retriever: embed query, fetch top-k from ChromaDB (with metadata)
# ─────────────────────────────────────────────────────────────────────────────

class ChromaRetrieverNode:
    """
    Embeds the query with OpenAI text-embedding-3-small and retrieves top-k chunks
    from ChromaDB. Returns chunks with text and metadata (filename, page) for source tracking.
    """

    def __init__(self):
        self._collection = None
        self._client = None

    def _ensure_client(self):
        if self._collection is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install: pip install openai") from e

        from src.ingest import get_chroma_collection
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            # Reuse the process-wide singleton — avoids a second Rust HNSW engine in RAM.
            _chroma_client, self._collection = get_chroma_collection()
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found. Run ingest first. Error: {e}"
            ) from e

    def _embed(self, text: str) -> list[float]:
        r = self._client.embeddings.create(input=[text], model=OPENAI_EMBEDDING_MODEL)
        return r.data[0].embedding

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        query = data["query"]
        mode = data.get("mode", "expert")
        k = data.get("k", TOP_K)

        self._ensure_client()
        variants = _retrieval_query_variants(query)
        log.info(
            f"[Retriever] Multi-query retrieval: {len(variants)} variant(s), top-{k} merged chunks..."
        )
        # Hämta fler per variant så sammanslagning kan välja bäst unika träffar.
        per_query = min(max(k, 12), 40)
        merged_rows: list[tuple[float, str, dict]] = []
        for v in variants:
            emb = self._embed(v)
            results = self._collection.query(
                query_embeddings=[emb],
                n_results=per_query,
                include=["documents", "metadatas", "distances"],
            )
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, dists):
                try:
                    d = float(dist)
                except (TypeError, ValueError):
                    d = 0.0
                merged_rows.append((d, doc or "", meta or {}))

        merged_rows.sort(key=lambda x: x[0])
        seen_keys: set[tuple[str, str, int]] = set()
        chunks: list[dict] = []
        for dist, doc, meta in merged_rows:
            key = _chunk_dedupe_key(doc, meta)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            chunks.append({"text": doc, "metadata": meta})
            if len(chunks) >= k:
                break

        log.info(f"[Retriever] Retrieved {len(chunks)} chunks (merged, deduplicated).")
        return {"query": query, "chunks": chunks, "mode": mode}


# ─────────────────────────────────────────────────────────────────────────────
# Generator: build prompt (Expert or Citizen), call OpenAI, append sources
# ─────────────────────────────────────────────────────────────────────────────

def _format_sources(chunks: list[dict]) -> str:
    """Build 'Källor: filename (sid N), ...' from chunks with metadata."""
    seen: set[tuple[str, int]] = set()
    parts = []
    for c in chunks:
        meta = c.get("metadata") or {}
        fn = meta.get("filename") or "okänd"
        page = meta.get("page")
        if page is None:
            page = 0
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = 0
        key = (fn, page)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{fn} (sid {page})")
    if not parts:
        return ""
    return "Källor: " + ", ".join(parts)


def _build_context_block(
    chunks: list[dict],
    *,
    max_chunks: int | None = None,
    max_context_chars: int | None = None,
    max_chunk_chars: int | None = None,
) -> tuple[str, list[dict]]:
    """
    Build a context block under a rough size budget.

    We don't have token counting here, so we cap by characters to avoid
    overlong prompts that reduce answer quality on large documents.
    """
    if not chunks:
        return "Ingen kontext hittades.", []

    max_chunks = max_chunks if max_chunks is not None else PROMPT_MAX_CHUNKS
    max_context_chars = max_context_chars if max_context_chars is not None else PROMPT_MAX_CONTEXT_CHARS
    max_chunk_chars = max_chunk_chars if max_chunk_chars is not None else PROMPT_MAX_CHUNK_CHARS

    parts: list[str] = []
    total_chars = 0
    selected_chunks: list[dict] = []

    for c in chunks[:max_chunks]:
        t = str(c.get("text") or "").strip()
        if not t:
            continue
        truncated_t = t
        if len(truncated_t) > max_chunk_chars:
            truncated_t = truncated_t[:max_chunk_chars] + "..."
        # +2 for the "\n\n" join between chunks.
        projected = total_chars + len(truncated_t) + (2 if parts else 0)
        if parts and projected > max_context_chars:
            break
        parts.append(truncated_t)
        selected_chunks.append(c)
        total_chars += len(truncated_t) + (2 if len(parts) > 1 else 0)

        if total_chars >= max_context_chars:
            break

    return (
        ("\n\n".join(parts) if parts else "Ingen kontext hittades."),
        selected_chunks,
    )


class GeneratorNode:
    """
    Uses OpenAI gpt-4o-mini with either Expert (Prompt A) or Citizen (Prompt B) system prompt.
    Appends strict source tracking to the answer: "Källor: ..."
    """

    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai not installed. Run: pip install openai") from e
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Support both direct retriever output and DagPipeline-wrapped (retriever_result)
        if "retriever_result" in data:
            data = data["retriever_result"]
        query = data["query"]
        chunks = data.get("chunks") or []
        mode = data.get("mode", "expert")

        if mode not in SYSTEM_PROMPTS:
            raise ValueError(f"Unknown mode '{mode}'. Use 'expert' or 'citizen'.")

        self._ensure_client()
        system_prompt = SYSTEM_PROMPTS[mode]
        context_block, selected_chunks = _build_context_block(chunks)

        user_content = (
            f"Underlag (utdrag ur indexerade dokument):\n\n{context_block}\n\n"
            f"Fråga: {query}\n\n"
            f"{RAG_USER_INSTRUCTIONS}\n\n"
            "Om underlaget innehåller både bakgrund (definitioner, metod, områdesbeskrivning) och "
            "saknar plats-/fastighetsspecifika värden: ta med bakgrunden i svaret, inte bara att "
            "siffror för en enskild fastighet saknas.\n"
            "Om underlaget beskriver stråk eller linjeföring (var banan går på bank/bro/tunnel, "
            "med platser i texten): återge den beskrivningen — det räknas som svar på var utformningen "
            "sker, även om inte varje fastighet näms.\n"
            "Om underlaget innehåller meningar om stråk, 'på bank', eller namngivna orter/vattendrag: "
            "de ska med i svaret. Skapa inte en motsägande slutparagraf om att lägen 'inte kan fastställas' "
            "för hela sträckan om sådan beskrivning redan finns ovan.\n"
            "Om svaret redan beskriver stråk och var bank förekommer: avsluta inte med att "
            "'detaljer om exakt lokalisering saknas' — då har du redan svarat på lokalisering på stråksnivå.\n"
            "Om frågan handlar om hur fastigheter eller närboende påverkas: prioritera utdrag om "
            "mark, buller, vibration, zoner och skyddsåtgärder — undvik att fylla med hastighet, "
            "restid och stationslista om det inte behövs för att svara.\n"
            "Om underlaget inte räcker för en viss detalj: säg vad som saknas och besvara i övrigt "
            "endast det som underlaget faktiskt stödjer."
        )

        log.info(f"[Generator] Generating answer in '{mode}' mode...")
        response = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        answer = (response.choices[0].message.content or "").strip()

        # Ensure source line is present; if model omitted it, append ourselves
        sources_line = _format_sources(selected_chunks)
        if sources_line and "Källor:" not in answer:
            answer = answer.rstrip() + "\n\n" + sources_line

        return {
            "answer": answer,
            "chunks": selected_chunks,
            "mode": mode,
            "query": query,
            "sources": sources_line,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DagPipeline assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_dag_pipeline():
    """Build Datapizza DagPipeline: retriever → generator."""
    try:
        from datapizza.pipeline import DagPipeline
    except ImportError as e:
        raise RuntimeError("datapizza-ai not installed. Run: pip install datapizza-ai") from e

    pipeline = DagPipeline()
    pipeline.add_module("retriever", ChromaRetrieverNode())
    pipeline.add_module("generator", GeneratorNode())
    pipeline.connect("retriever", "generator", target_key="retriever_result")

    return pipeline


@lru_cache(maxsize=1)
def _get_rag_nodes() -> tuple[ChromaRetrieverNode, GeneratorNode]:
    """
    Return a single shared (retriever, generator) pair for the process lifetime.
    lru_cache ensures the ChromaDB connection and OpenAI client are created once
    and reused across every query, instead of being rebuilt on every call.
    """
    return ChromaRetrieverNode(), GeneratorNode()


def run_rag(query: str, mode: str = "expert", k: int | None = None) -> dict[str, Any]:
    """
    Run the full RAG pipeline (retriever → generator). Uses DagPipeline nodes;
    execution is sequential so output is deterministic and compatible with all DagPipeline versions.

    Parameters
    ----------
    query : str
        User question (Swedish).
    mode : str
        "expert" (Prompt A) or "citizen" (Prompt B).
    k : int, optional
        Top-k chunks to retrieve (default from config).

    Returns
    -------
    dict with "answer", "chunks", "mode", "query", "sources".
    """
    run_k = k if k is not None else TOP_K
    retriever, generator = _get_rag_nodes()
    retriever_out = retriever({"query": query, "mode": mode, "k": run_k})
    gen_out = generator(retriever_out)
    return gen_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Vilka miljökrav gäller?"
    print("\n=== EXPERT ===")
    out = run_rag(q, mode="expert")
    print(out["answer"])
    print("\n=== CITIZEN ===")
    out = run_rag(q, mode="citizen")
    print(out["answer"])
