"""
Loads the curated knowledge base and variable-interaction graph from
/data (the project's structured-dataset layer, per hackathon requirement
1: "Knowledge System (Critical) ... structured datasets").
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

# /data lives at the repo root, two levels above backend/app/knowledge/
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
KB_PATH = DATA_DIR / "knowledge_base.json"
EDGES_PATH = DATA_DIR / "variable_interactions.json"


@lru_cache(maxsize=1)
def load_documents() -> List[Dict[str, Any]]:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["documents"]


@lru_cache(maxsize=1)
def load_interaction_graph() -> Dict[str, Any]:
    with open(EDGES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_document_by_id(kb_id: str) -> Dict[str, Any] | None:
    for doc in load_documents():
        if doc["id"] == kb_id:
            return doc
    return None


def documents_as_corpus_text() -> List[str]:
    """
    Builds the retrieval text for each document: title + mechanism +
    finding_summary + variables studied. This is what gets embedded/chunked.
    We deliberately embed the mechanism + finding together (not the raw
    'notes' or 'evidence_notes' fields) so retrieval matches on scientific
    content, not editorial commentary.
    """
    texts = []
    for doc in load_documents():
        parts = [
            doc["title"],
            doc.get("mechanism", ""),
            doc.get("finding_summary", ""),
            "Variables: " + ", ".join(doc.get("variables_studied", [])),
            "Domain: " + ", ".join(doc.get("domain", [])),
            "Region: " + doc.get("region", ""),
            "Ecosystem: " + doc.get("ecosystem", ""),
        ]
        texts.append(". ".join(p for p in parts if p))
    return texts
