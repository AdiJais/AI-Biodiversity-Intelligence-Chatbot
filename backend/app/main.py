"""
Darukaa.Earth AI Biodiversity Intelligence API.

Endpoints:
  GET  /health            liveness + which retrieval backend is active
  POST /chat               free-text (+ optional structured) conversational turn
  POST /assess              pure structured-JSON environmental assessment, no text needed
  GET  /knowledge/{kb_id}   inspect a single knowledge base source (transparency/debug)
  GET  /knowledge           list all knowledge base sources (transparency/debug)

Run locally:
  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.knowledge.loader import load_documents, get_document_by_id
from app.models.schemas import ChatRequest, EnvironmentalObservation, StructuredResponse
from app.pipeline import run_pipeline
from app.rag.retriever import backend_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("darukaa")

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence API",
    description="Evidence-grounded, multi-metric environmental reasoning system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon demo setting; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "retrieval_backend": backend_name(),
        "knowledge_base_size": len(load_documents()),
    }


@app.post("/chat", response_model=StructuredResponse)
def chat(request: ChatRequest):
    if not request.message and not request.structured:
        raise HTTPException(status_code=400, detail="Provide 'message' text and/or 'structured' data.")
    try:
        return run_pipeline(request)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Internal reasoning error: {exc}")


class AssessRequest(BaseModel):
    session_id: str
    observation: EnvironmentalObservation


@app.post("/assess", response_model=StructuredResponse)
def assess(request: AssessRequest):
    """Pure structured-input endpoint (bonus requirement: structured JSON input)."""
    chat_request = ChatRequest(session_id=request.session_id, message="", structured=request.observation)
    try:
        return run_pipeline(chat_request)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Internal reasoning error: {exc}")


@app.get("/knowledge")
def list_knowledge():
    docs = load_documents()
    return [
        {"id": d["id"], "title": d["title"], "year": d.get("year"), "evidence_type": d.get("evidence_type")}
        for d in docs
    ]


@app.get("/knowledge/{kb_id}")
def get_knowledge(kb_id: str):
    doc = get_document_by_id(kb_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"No knowledge base entry '{kb_id}'")
    return doc
