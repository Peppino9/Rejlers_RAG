"""
app.py — Streamlit frontend for Rejlers RAG PoC
----------------------------------------------
- Modern dark UI.
- Drag & drop PDFs to ingest to ChromaDB.
- List/remove documents from the database.
- Ask a Swedish question with Expert vs Citizen mode.
- Displays: answer, sources, LIX, Ragas Faithfulness & Answer Relevance.
- Expander: raw retrieved chunks for expert verification.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

# Ensure project root and src are on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from config import OPENAI_API_KEY
from src.ingest import (
    clear_collection,
    delete_by_filename,
    get_chroma_collection,
    ingest_pdf_bytes,
    ingest_storymaps_sections,
    list_indexed_filenames,
    run_ingestion,
)
from src.pipeline import run_rag
from src.evaluate import lix_score, ragas_evaluate


STRAK_REFERENCE_IMAGE = Path(__file__).resolve().parent / "assets" / "strak-reference.png"
STRAK_KEYWORDS = (
    "strak",
    "stråk",
    "morkbla",
    "mörkblå",
    "morkblå",
    "ljusbla",
    "ljusblå",
    "bla",
    "blå",
    "brun",
    "gron",
    "grön",
    "lila",
    "rod",
    "röd",
    "orange",
)


def _norm_text(s: str) -> str:
    """Lowercase + remove common swedish diacritics for robust keyword matching."""
    return (
        (s or "")
        .lower()
        .replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
    )


def _should_show_strak_reference(query: str, answer: str, chunks: list[dict]) -> bool:
    """Show map when user asks about stråk/colors, or when retrieved context mentions it."""
    texts = [query or "", answer or ""]
    for c in chunks or []:
        if not isinstance(c, dict):
            continue
        texts.append(str(c.get("text", "") or ""))
    haystack = _norm_text("\n".join(texts))
    return any(re.search(rf"\b{re.escape(k)}\b", haystack) for k in STRAK_KEYWORDS)


def _plain_excerpt(text: str, max_chars: int = 420) -> str:
    """Plain text excerpt for källhänvisningar (no word highlighting)."""
    raw = (text or "").strip()
    if not raw:
        return "Ingen text tillgänglig för denna källa."
    return raw[:max_chars] + ("…" if len(raw) > max_chars else "")




st.set_page_config(
    page_title="Rejlers RAG — LIX vs Faktagrund",
    page_icon="🗂️",
    layout="wide",
)

st.markdown(
    """
    <style>
      :root {
        --bg0:#0b0d13; --bg1:#111629; --bg2:#151b33; --fg0:#e8ecff; --fg1:#aab4e6;
        --acc:#7c5cff; --acc2:#25d6b5; --warn:#ffcc66; --danger:#ff5c7a;
        --card: rgba(255,255,255,0.04); --card2: rgba(255,255,255,0.06);
        --border: rgba(255,255,255,0.10);
      }
      .stApp { background: radial-gradient(1000px 600px at 20% 10%, rgba(124,92,255,0.22), transparent 60%),
                       radial-gradient(900px 500px at 75% 30%, rgba(37,214,181,0.14), transparent 60%),
                       linear-gradient(180deg, var(--bg0), var(--bg1)); color: var(--fg0); }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .subtle { color: var(--fg1); }
      .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 16px 16px; }
      .pill { display:inline-block; padding: 6px 10px; border-radius: 999px; background: rgba(124,92,255,0.16);
              border: 1px solid rgba(124,92,255,0.35); color: var(--fg0); font-size: 12px; }
      .muted { color: var(--fg1); }
      .hr { height:1px; background: rgba(255,255,255,0.10); margin: 12px 0; }
      .danger { color: var(--danger); }
      .stButton button { border-radius: 12px; border: 1px solid rgba(255,255,255,0.14); }
      .stTextArea textarea, .stTextInput input { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Rejlers Multi-Document RAG")
st.markdown(
    "<span class='subtle'>PoC för examensarbete: Läsbarhet (LIX) vs faktagrundad noggrannhet</span>",
    unsafe_allow_html=True,
)

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY saknas. Lägg den i en .env-fil i projektroten.")
    st.stop()

with st.sidebar:
    st.markdown("### Dokumentdatabas")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    try:
        _client, collection = get_chroma_collection()
        st.markdown(f"**Chunks i DB**: `{collection.count()}`")
    except Exception as e:
        st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown("**PDF:er**")
    st.caption("Indexerar alla PDF:er i `./data` — samma som `python -m src.ingest`.")
    if st.button("Indexera PDF:er från ./data", use_container_width=True):
        with st.spinner("Indexerar (Docling → chunk → embeddings → ChromaDB)..."):
            try:
                run_ingestion()
            except Exception as e:
                st.exception(e)
            import gc
            gc.collect()
            try:
                import torch
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass
        st.success("Klart. Alla PDF:er i ./data är indexerade.")
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("**Ladda upp ny PDF**")
    st.caption("För PDF:er som inte ligger i `./data`.")
    uploaded = st.file_uploader(
        "Dra & släpp PDF-filer här",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Indexera uppladdade PDF:er", use_container_width=True, disabled=not uploaded):
        total_added = 0
        with st.spinner("Indexerar (Docling → chunk → batch-embeddings → ChromaDB)..."):
            for uf in uploaded or []:
                try:
                    added = ingest_pdf_bytes(uf.getvalue(), filename=uf.name)
                    total_added += int(added)
                except Exception as e:
                    st.error(f"Misslyckades: `{uf.name}` — {e}")
            import gc
            gc.collect()
            try:
                import torch
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass
        st.success(f"Klart. Lagt till `{total_added}` chunks.")
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("**StoryMaps (scrapad text)**")
    st.caption("Indexerar `data/sweco_storymaps_*_item_*/sections.md` till samma databas som PDF:er.")
    if st.button("Indexera StoryMaps markdown", use_container_width=True):
        try:
            with st.spinner("Läser sections.md → chunk → embeddings → ChromaDB..."):
                n = ingest_storymaps_sections()
            st.success(f"Klart. Lagt till/uppdaterat `{n}` chunks från StoryMaps.")
        except Exception as e:
            st.exception(e)
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("**Hantera dokument**")
    filenames = list_indexed_filenames()
    if filenames:
        selected = st.selectbox("Välj dokument", filenames, label_visibility="collapsed")
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Ta bort dokument", use_container_width=True):
                deleted = delete_by_filename(selected)
                st.warning(f"Tog bort `{deleted}` chunks för `{selected}`.")
                st.rerun()
        with cols[1]:
            confirm = st.checkbox("Jag förstår (rensa allt)", value=False)
            if st.button("Rensa hela databasen", use_container_width=True, disabled=not confirm):
                deleted = clear_collection()
                st.warning(f"Databasen rensad. Tog bort `{deleted}` chunks.")
                st.rerun()
    else:
        st.info("Inga dokument indexerade ännu.")

    st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.markdown("### Fråga")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    query = st.text_area(
        "Fråga (svenska)",
        placeholder="T.ex. Vilka bullerkrav gäller för byggnad X? Ange gärna plats/objekt och vad du vill veta.",
        height=140,
        label_visibility="collapsed",
    )
    mode = st.radio(
        "Genereringsläge",
        options=["expert", "citizen"],
        format_func=lambda x: "Expert (Prompt A) — teknisk, exakt" if x == "expert" else "Medborgare (Prompt B) — förenklat, lågt LIX",
        horizontal=True,
    )
    submit = st.button("Sök och generera", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("### Tips för bättre träffar")
    st.markdown(
        """
        <div class='card'>
          <div class='pill'>Sökprecision</div>
          <div class='hr'></div>
          <div class='muted'>
            Skriv gärna <b>objekt</b> (bro/tunnel/område), <b>disciplin</b> (buller, vibration, geoteknik)
            och <b>kravtyp</b> (gränsvärde, metod, tidsperiod).<br/><br/>
            Om du vet ett dokumentnamn, skriv det i frågan.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if submit and query.strip():
    with st.spinner("Hämtar kontext och genererar svar..."):
        try:
            out = run_rag(query.strip(), mode=mode)
        except Exception as e:
            st.exception(e)
            st.stop()

    answer = out.get("answer", "")
    chunks = out.get("chunks", [])
    sources = out.get("sources", "")

    st.markdown("### Genererat svar")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(answer if answer else "_Inget svar._")
    if _should_show_strak_reference(query.strip(), answer, chunks) and STRAK_REFERENCE_IMAGE.exists():
        st.markdown("**Referensbild: utredda stråk**")
        st.image(str(STRAK_REFERENCE_IMAGE), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Källhänvisningar")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.text(sources if sources else "Inga källor returnerade.")

    seen_sources: set[tuple[str, int]] = set()
    for i, c in enumerate(chunks or []):
        meta = c.get("metadata") or {}
        fn = str(meta.get("filename") or "okänd")
        page_raw = meta.get("page", 0)
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            page = 0
        source_key = (fn, page)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        with st.expander(f"{fn} (sid {page})", expanded=(i == 0)):
            st.text(_plain_excerpt(c.get("text", "")))
    st.markdown("</div>", unsafe_allow_html=True)

    # Optional: render relevant images from retrieved asset chunks.
    # These are produced by src/ingest_storymaps_assets.py (stored as asset_type=image).
    asset_images: list[tuple[str, str]] = []  # (asset_path, caption)
    for c in chunks or []:
        meta = c.get("metadata") or {}
        if meta.get("asset_type") != "image":
            continue
        asset_path = meta.get("asset_path")
        if not isinstance(asset_path, str) or not asset_path:
            continue
        caption = meta.get("caption") or meta.get("alt") or ""
        caption = str(caption)
        asset_images.append((asset_path, caption))

    # Deduplicate (keep order).
    dedup: dict[str, str] = {}
    for p, cap in asset_images:
        if p not in dedup:
            dedup[p] = cap
    asset_images = [(p, dedup[p]) for p in dedup]

    if asset_images:
        st.markdown("### Bilder från hämtad kontext")
        cols = st.columns(min(3, len(asset_images)))
        for idx, (p, cap) in enumerate(asset_images[:3]):
            with cols[idx % 3]:
                try:
                    st.image(p, caption=cap[:140] if cap else None, width="stretch")
                except Exception:
                    st.write(f"_Kunde inte visa bild:_ `{p}`")

    # Metrics
    st.markdown("### Vetenskapliga mått")
    chunk_texts = [c.get("text", "") for c in chunks if c.get("text")]
    lix = lix_score(answer)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LIX (läsbarhet)", f"{lix:.2f}")

    run_ragas = st.checkbox(
        "Kör Ragas-utvärdering (Faithfulness & Answer Relevance)",
        value=False,
        help="Ragas laddar tunga ML-modeller första gången — aktivera manuellt vid behov.",
    )
    if run_ragas:
        with st.spinner("Kör Ragas..."):
            try:
                ragas = ragas_evaluate(query.strip(), answer, chunk_texts)
                with col2:
                    st.metric("Ragas Faithfulness", f"{ragas['faithfulness']:.3f}")
                with col3:
                    st.metric("Ragas Answer Relevance", f"{ragas['answer_relevancy']:.3f}")
            except Exception as e:
                with col2:
                    st.warning("Ragas: " + str(e))
                with col3:
                    st.write("—")
    else:
        with col2:
            st.metric("Ragas Faithfulness", "—")
        with col3:
            st.metric("Ragas Answer Relevance", "—")

    # Raw chunks for experts
    with st.expander("Visa hämtad kontext (råa chunkar) — för expertgranskning"):
        for i, c in enumerate(chunks):
            meta = c.get("metadata") or {}
            fn = meta.get("filename", "?")
            page = meta.get("page", "?")
            st.markdown(f"**Chunk {i+1}** — `{fn}` (sid {page})")
            st.text(c.get("text", "")[:2000] + ("…" if len(c.get("text", "")) > 2000 else ""))
            st.divider()

elif submit and not query.strip():
    st.warning("Skriv en fråga först.")
