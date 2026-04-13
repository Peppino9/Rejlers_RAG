"""
app.py — Streamlit frontend with a modern UI and local Q/A history.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import OPENAI_API_KEY
from src.evaluate import lix_score, ragas_evaluate
from src.pipeline import run_rag

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_response" not in st.session_state:
        st.session_state.current_response = {
            "question": "",
            "answer": "Ask your first question.",
            "sources": "",
            "timestamp": "",
            "metrics": {"lix": None, "faithfulness": None, "answer_relevancy": None},
        }
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""


def _fmt_metric(value: object, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _save_response(question: str, answer: str, sources: str, metrics: dict) -> None:
    item = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "timestamp": datetime.now().strftime("%H:%M"),
        "metrics": metrics,
    }
    st.session_state.current_response = item
    st.session_state.history.append(item)
    st.session_state.history = st.session_state.history[-30:]


st.set_page_config(page_title="ALMA × Rejlers", page_icon="✦", layout="wide")
_init_state()

project_root = Path(__file__).resolve().parent
alma_logo = project_root / "assets" / "alma_logo.png"
rejlers_logo_candidates = [
    project_root / "assets" / "rejlers_logo.png",
    project_root / "assets" / "rejlers-logo.png",
    project_root / "assets" / "rejlers.png",
]
rejlers_logo = next((p for p in rejlers_logo_candidates if p.exists()), None)

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY saknas. Lägg den i en .env-fil i projektroten.")
    st.stop()

st.markdown(
    """
    <style>
      :root {
        --text: #e8ecff;
        --muted: #b8c0dd;
        --border: rgba(255,255,255,0.16);
        --card: rgba(255,255,255,0.08);
        --card-2: rgba(255,255,255,0.10);
      }
      .stApp {
        color: var(--text);
        background:
          radial-gradient(900px 500px at 12% 12%, rgba(77, 118, 255, 0.34), transparent 65%),
          radial-gradient(900px 600px at 92% 82%, rgba(47, 210, 180, 0.24), transparent 65%),
          radial-gradient(650px 420px at 50% 100%, rgba(188, 72, 255, 0.18), transparent 72%),
          linear-gradient(160deg, #070b18 0%, #0f1630 52%, #0a1227 100%);
      }
      .main .block-container { max-width: 1120px; padding-top: 1.7rem; }
      [data-testid="stSidebar"] > div:first-child {
        background: rgba(8, 11, 24, 0.72);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-right: 1px solid var(--border);
      }
      .hero-title {
        margin: 0 0 14px 0;
        font-size: 4.1rem;
        line-height: 0.98;
        letter-spacing: -0.04em;
        font-weight: 720;
      }
      .subline {
        margin: 0 0 14px 0;
        color: var(--muted);
        font-size: 1.02rem;
      }
      .chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
      .chip {
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.12);
        color: #dbe3ff;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 0.88rem;
      }
      .composer {
        border: 1px solid var(--border);
        background: linear-gradient(145deg, var(--card-2), var(--card));
        border-radius: 26px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 12px;
      }
      .answer-card {
        margin-top: 14px;
        border: 1px solid var(--border);
        background: linear-gradient(145deg, rgba(255,255,255,0.11), rgba(255,255,255,0.07));
        border-radius: 20px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 18px;
      }
      .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
      }
      .metric-pill {
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.12);
        color: #eef2ff;
        padding: 6px 11px;
        font-size: 0.84rem;
      }
      .tiny { color: var(--muted); font-size: 0.83rem; }
      .stTextArea textarea {
        min-height: 220px !important;
        color: #eef2ff !important;
        background: rgba(7, 11, 25, 0.44) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: 18px !important;
        font-size: 1.03rem !important;
      }
      .stButton button {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        background: rgba(255,255,255,0.14) !important;
        color: #f4f7ff !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    if alma_logo.exists():
        st.image(str(alma_logo), width=140)
    st.caption("ALMA")
    if rejlers_logo:
        st.image(str(rejlers_logo), width=140)
        st.caption("Rejlers")
    else:
        st.markdown("**Rejlers**")
    st.markdown("<p class='tiny'>Temporary local history only</p>", unsafe_allow_html=True)

    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_response = {
            "question": "",
            "answer": "History cleared. Ask a new question.",
            "sources": "",
            "timestamp": "",
            "metrics": {"lix": None, "faithfulness": None, "answer_relevancy": None},
        }
        st.session_state.query_input = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### Previous Q/A")
    for idx in range(len(st.session_state.history) - 1, -1, -1):
        item = st.session_state.history[idx]
        label = f"{item['timestamp']} · {item['question'][:34] or 'Question'}"
        if st.button(label, key=f"history_{idx}", use_container_width=True):
            st.session_state.current_response = item
            st.session_state.query_input = item["question"]
            st.rerun()

st.markdown("<h1 class='hero-title'>Welcome to ALMA.</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subline'>Ask one focused question. Get one grounded answer.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="chip-row">
      <div class="chip">Permit summary</div>
      <div class="chip">Infrastructure constraints</div>
      <div class="chip">Buller and vibration</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='composer'>", unsafe_allow_html=True)
query = st.text_area(
    "Question input",
    key="query_input",
    label_visibility="collapsed",
    placeholder="What do you want to know?",
)
ask = st.button("Generate answer", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if ask:
    question = query.strip()
    if not question:
        st.warning("Write a question first.")
    else:
        with st.spinner("Generating answer..."):
            try:
                out = run_rag(question, mode="citizen")
                answer = out.get("answer", "").strip() or "No answer returned."
                sources = out.get("sources", "")
                chunks = out.get("chunks", [])
                chunk_texts = [c.get("text", "") for c in chunks if isinstance(c, dict) and c.get("text")]
                metrics = {
                    "lix": lix_score(answer),
                    "faithfulness": None,
                    "answer_relevancy": None,
                }
                try:
                    ragas = ragas_evaluate(question, answer, chunk_texts)
                    metrics["faithfulness"] = ragas.get("faithfulness")
                    metrics["answer_relevancy"] = ragas.get("answer_relevancy")
                except Exception:
                    pass
                _save_response(question, answer, sources, metrics)
            except Exception as exc:
                st.error(f"Request failed: {exc}")

response = st.session_state.current_response
metrics = response.get("metrics", {})

st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
st.markdown("### Answer")
st.markdown(response.get("answer", "No answer yet."))
if response.get("sources"):
    st.markdown("<p class='tiny'><b>Sources</b></p>", unsafe_allow_html=True)
    st.caption(response["sources"])
st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-pill">LIX: {_fmt_metric(metrics.get('lix'), 2)}</div>
      <div class="metric-pill">RAGAS Faithfulness: {_fmt_metric(metrics.get('faithfulness'))}</div>
      <div class="metric-pill">RAGAS Relevance: {_fmt_metric(metrics.get('answer_relevancy'))}</div>
      <div class="metric-pill">Saved: {response.get('timestamp') or '--'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
