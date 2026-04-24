"""
FastAPI backend for the React frontend.

Run:
  uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["expert", "citizen"] = "citizen"


class AskResponse(BaseModel):
    answer: str
    sources: str
    metrics: dict[str, float | None]


app = FastAPI(title="Rejlers RAG API", version="1.0.0")

_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5180",
    "http://127.0.0.1:5180",
]
_env_origins = [
    o.strip() for o in os.getenv("FRONTEND_ORIGIN", "").split(",") if o.strip()
]
_allow_origins = sorted(set(_default_origins + _env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    # Local dev + any Railway public HTTPS app (so the React site can call the API without CORS errors).
    allow_origin_regex=(
        r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
        r"|https://[a-zA-Z0-9-]+\.up\.railway\.app"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Rejlers RAG API",
        "health": "/health",
        "ask": "POST /api/ask",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    # Import here so /health can respond while heavy deps load on first question.
    from src.evaluate import lix_score, ragas_evaluate_stable
    from src.pipeline import run_rag

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        out = run_rag(question, mode=payload.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {exc}") from exc

    answer = (out.get("answer") or "").strip() or "No answer returned."
    sources = out.get("sources") or ""
    chunks = out.get("chunks") or []
    chunk_texts = [c.get("text", "") for c in chunks if isinstance(c, dict) and c.get("text")]

    metrics: dict[str, float | None] = {
        "lix": float(lix_score(answer)),
        "faithfulness": None,
        "faithfulness_std": None,
        "answer_relevancy": None,
        "answer_relevancy_std": None,
        "answer_relevancy_runs": None,
        "lexical_relevance": None,
    }
    try:
        ragas = ragas_evaluate_stable(question, answer, chunk_texts)
        faithfulness = ragas.get("faithfulness")
        faithfulness_std = ragas.get("faithfulness_std")
        answer_relevancy = ragas.get("answer_relevancy")
        answer_relevancy_std = ragas.get("answer_relevancy_std")
        answer_relevancy_runs = ragas.get("answer_relevancy_runs")
        lexical_relevance = ragas.get("lexical_relevance")
        metrics["faithfulness"] = float(faithfulness) if faithfulness is not None else None
        metrics["faithfulness_std"] = float(faithfulness_std) if faithfulness_std is not None else None
        metrics["answer_relevancy"] = (
            float(answer_relevancy) if answer_relevancy is not None else None
        )
        metrics["answer_relevancy_std"] = (
            float(answer_relevancy_std) if answer_relevancy_std is not None else None
        )
        metrics["answer_relevancy_runs"] = (
            float(answer_relevancy_runs) if answer_relevancy_runs is not None else None
        )
        metrics["lexical_relevance"] = (
            float(lexical_relevance) if lexical_relevance is not None else None
        )
    except Exception:
        # If Ragas models fail to load or timeout, still return answer + LIX.
        pass

    return AskResponse(answer=answer, sources=sources, metrics=metrics)
