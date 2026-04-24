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

RAGAS_RELEVANCY_RUNS: int = 3
_LEXICAL_STOPWORDS: frozenset[str] = frozenset(
    {
        "och", "att", "som", "för", "med", "till", "inte", "kan", "har", "den", "det",
        "detta", "dessa", "är", "var", "blir", "från", "vid", "per", "inom",
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
    }
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


def _safe_metric_float(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def _tokens_for_lexical_relevance(text: str) -> set[str]:
    words = re.findall(r"[A-Za-zÅÄÖåäö0-9\-]{3,}", (text or "").lower())
    return {w for w in words if w not in _LEXICAL_STOPWORDS}


def lexical_relevance_score(question: str, answer: str) -> float | None:
    """
    Lightweight lexical fallback: fraction of question keywords seen in answer.
    Returns None if there are no usable question keywords.
    """
    q_tokens = _tokens_for_lexical_relevance(question)
    if not q_tokens:
        return None
    a_tokens = _tokens_for_lexical_relevance(answer)
    if not a_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens) / len(q_tokens)
    return round(max(0.0, min(1.0, overlap)), 3)


def _mean_and_std(values: List[float | None]) -> tuple[float | None, float | None]:
    usable = [v for v in values if v is not None]
    if not usable:
        return None, None
    mean = sum(usable) / len(usable)
    if len(usable) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in usable) / len(usable)
    return mean, math.sqrt(variance)


def ragas_evaluate(
    question: str,
    answer: str,
    contexts: List[List[str]] | List[str],
) -> dict[str, float | None]:
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
    Metric values are None when the metric could not be computed.
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
        return {"faithfulness": None, "answer_relevancy": None}
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
        return {"faithfulness": None, "answer_relevancy": None}

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
        return {"faithfulness": None, "answer_relevancy": None}

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
        return {"faithfulness": None, "answer_relevancy": None}

    df = result.to_pandas()
    if df.empty:
        log.warning("Ragas returned empty dataframe for question: %s", question[:200])
        return {"faithfulness": None, "answer_relevancy": None, "ragas_result": result}
    row = df.iloc[0]
    answer_rel = row.get("answer_relevancy", row.get("answer_relevance", 0.0))
    faithfulness_value = _safe_metric_float(row.get("faithfulness", 0.0))
    answer_relevancy_value = _safe_metric_float(answer_rel)
    if faithfulness_value is None or answer_relevancy_value is None:
        log.warning(
            "Ragas metric missing/NaN (faithfulness=%r, answer_relevancy=%r). "
            "question_len=%d answer_len=%d contexts=%d",
            row.get("faithfulness", None),
            answer_rel,
            len(question or ""),
            len(answer_judge),
            len(contexts_judge),
        )
    return {
        "faithfulness": faithfulness_value,
        "answer_relevancy": answer_relevancy_value,
        "ragas_result": result,
    }


def ragas_evaluate_stable(
    question: str,
    answer: str,
    contexts: List[List[str]] | List[str],
    runs: int = RAGAS_RELEVANCY_RUNS,
) -> dict[str, float | None]:
    """
    Run RAGAS multiple times and aggregate to reduce judge variance.
    """
    n_runs = max(1, int(runs))
    faithfulness_runs: List[float | None] = []
    relevancy_runs: List[float | None] = []
    last_result: object | None = None

    for _ in range(n_runs):
        out = ragas_evaluate(question, answer, contexts)
        faithfulness_runs.append(out.get("faithfulness"))  # type: ignore[arg-type]
        relevancy_runs.append(out.get("answer_relevancy"))  # type: ignore[arg-type]
        if "ragas_result" in out:
            last_result = out["ragas_result"]

    faithfulness_mean, faithfulness_std = _mean_and_std(faithfulness_runs)
    answer_relevancy_mean, answer_relevancy_std = _mean_and_std(relevancy_runs)
    lexical = lexical_relevance_score(question, answer)
    successful_runs = float(sum(1 for v in relevancy_runs if v is not None))

    return {
        "faithfulness": faithfulness_mean,
        "faithfulness_std": faithfulness_std,
        "answer_relevancy": answer_relevancy_mean,
        "answer_relevancy_std": answer_relevancy_std,
        "answer_relevancy_runs": successful_runs,
        "lexical_relevance": lexical,
        "ragas_result": last_result,  # optional debug object
    }


def compute_metrics(
    question: str,
    answer: str,
    contexts: List[str],
) -> dict[str, float | None]:
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
    RAGAS values may be None if the metric computation fails.
    """
    lix = lix_score(answer)
    ragas_out = ragas_evaluate_stable(question, answer, [contexts])
    return {
        "lix": lix,
        "faithfulness": ragas_out["faithfulness"],
        "answer_relevancy": ragas_out["answer_relevancy"],
        "answer_relevancy_std": ragas_out["answer_relevancy_std"],
        "answer_relevancy_runs": ragas_out["answer_relevancy_runs"],
        "lexical_relevance": ragas_out["lexical_relevance"],
    }
