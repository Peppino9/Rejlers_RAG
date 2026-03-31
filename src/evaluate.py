"""
src/evaluate.py — Science / Metrics
-----------------------------------
- Ragas: Faithfulness and Answer Relevance (using gpt-4o-mini as judge).
- Custom Swedish LIX (Läsbarhetsindex):
  LIX = (total_words / total_sentences) + (long_words * 100 / total_words)
  where long_words = words with more than 6 letters.

Ragas notes
-----------
Faithfulness builds a large JSON; long answers + huge contexts hit the default
max_tokens and break. We trim what we send to the judge and pass an explicit
LangChain ChatOpenAI with higher max_tokens plus OpenAIEmbeddings (fixes
embed_query compatibility with Ragas).
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    LLM_MODEL,
    RAGAS_JUDGE_MAX_ANSWER_CHARS,
    RAGAS_JUDGE_MAX_CHUNK_CHARS,
    RAGAS_JUDGE_MAX_CHUNKS,
    RAGAS_JUDGE_MAX_CONTEXT_CHARS,
    RAGAS_JUDGE_LLM_MAX_TOKENS,
)

# ─────────────────────────────────────────────────────────────────────────────
# LIX (Läsbarhetsindex) for Swedish
# ─────────────────────────────────────────────────────────────────────────────


def lix_score(text: str) -> float:
    """
    Compute Swedish LIX readability score.

    Formula: (total_words / total_sentences) + (long_words * 100 / total_words)
    Long word = word with more than 6 letters.

    Returns
    -------
    float
        LIX value. Lower = easier to read; ~25–30 = easy, 40+ = difficult.
    """
    if not text or not text.strip():
        return 0.0

    # Simple sentence split on . ! ?
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    total_sentences = max(len(sentences), 1)

    # Words: split on whitespace, strip punctuation for length
    words = re.findall(r"\b\w+\b", text)
    total_words = max(len(words), 1)
    long_words = sum(1 for w in words if len(w) > 6)

    lix = (total_words / total_sentences) + (long_words * 100 / total_words)
    return round(lix, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Ragas: shrink inputs + explicit LLM / embeddings
# ─────────────────────────────────────────────────────────────────────────────


def _strip_kallor_line(answer: str) -> str:
    """Remove trailing 'Källor: ...' so the judge scores the substantive answer."""
    lines = answer.strip().split("\n")
    while lines and lines[-1].strip().startswith("Källor:"):
        lines.pop()
    return "\n".join(lines).strip()


def _truncate_answer_for_judge(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _shrink_contexts_for_ragas(chunks: List[str]) -> List[str]:
    """Fewer / shorter chunks so faithfulness + relevancy stay within model limits."""
    out: List[str] = []
    budget = 0
    for c in chunks[:RAGAS_JUDGE_MAX_CHUNKS]:
        piece = (c or "").strip()
        if not piece:
            continue
        if len(piece) > RAGAS_JUDGE_MAX_CHUNK_CHARS:
            piece = piece[: RAGAS_JUDGE_MAX_CHUNK_CHARS] + "…"
        if budget + len(piece) > RAGAS_JUDGE_MAX_CONTEXT_CHARS:
            room = RAGAS_JUDGE_MAX_CONTEXT_CHARS - budget
            if room > 400:
                out.append(piece[:room] + "…")
            break
        out.append(piece)
        budget += len(piece)
    return out if out else [""]


@lru_cache(maxsize=1)
def _ragas_llm_and_embeddings():
    """Single cached judge LLM + embeddings (OpenAI-compatible, embed_query OK)."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set for Ragas metrics.")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=RAGAS_JUDGE_LLM_MAX_TOKENS,
    )
    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    return llm, embeddings


def _safe_metric_float(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    return x


def ragas_evaluate(
    question: str,
    answer: str,
    contexts: List[List[str]] | List[str],
) -> dict[str, float]:
    """
    Compute Ragas Faithfulness and Answer Relevance for one Q&A pair.

    Parameters
    ----------
    question : str
        User question.
    answer : str
        Model-generated answer.
    contexts : list of list of str, or list of str
        Retrieved context chunks. Ragas expects list[list[str]]; if list[str],
        wrapped as [contexts].

    Returns
    -------
    dict with keys "faithfulness", "answer_relevancy", and optionally "ragas_result".
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.run_config import RunConfig
    except ImportError as e:
        raise RuntimeError(
            "ragas/datasets not installed. Run: pip install ragas datasets"
        ) from e

    if not contexts:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    if isinstance(contexts[0], str):
        flat: List[str] = contexts  # type: ignore[assignment]
    else:
        flat = [c for group in contexts for c in (group or [])]

    answer_judge = _truncate_answer_for_judge(
        _strip_kallor_line(answer),
        RAGAS_JUDGE_MAX_ANSWER_CHARS,
    )
    contexts_judge = _shrink_contexts_for_ragas(flat)

    if not answer_judge:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

    data = {
        "question": [question],
        "answer": [answer_judge],
        "contexts": [contexts_judge],
    }
    dataset = Dataset.from_dict(data)

    try:
        llm, embeddings = _ragas_llm_and_embeddings()
    except Exception as e:
        log.warning("Ragas LLM/embeddings init failed: %s", e)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(timeout=300, max_retries=6),
            show_progress=False,
            raise_exceptions=False,
        )
    except Exception as e:
        log.warning("Ragas evaluate failed: %s", e)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

    df = result.to_pandas()
    if df.empty:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "ragas_result": result}
    row = df.iloc[0]
    answer_rel = row.get("answer_relevancy", row.get("answer_relevance", 0.0))
    return {
        "faithfulness": _safe_metric_float(row.get("faithfulness", 0.0)),
        "answer_relevancy": _safe_metric_float(answer_rel),
        "ragas_result": result,
    }


def compute_metrics(
    question: str,
    answer: str,
    contexts: List[str],
) -> dict[str, float]:
    """
    One-shot metrics for the Streamlit app: LIX + Ragas Faithfulness + Answer Relevance.

    Parameters
    ----------
    question : str
        User question.
    answer : str
        Generated answer.
    contexts : list of str
        Retrieved chunk texts.

    Returns
    -------
    dict with "lix", "faithfulness", "answer_relevancy".
    """
    lix = lix_score(answer)
    ragas_out = ragas_evaluate(question, answer, [contexts])
    return {
        "lix": lix,
        "faithfulness": ragas_out["faithfulness"],
        "answer_relevancy": ragas_out["answer_relevancy"],
    }
