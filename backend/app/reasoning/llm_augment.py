"""
Optional LLM augmentation (Agentic RAG differentiator - P1, optional).

This module is intentionally isolated and feature-flagged by environment
variable only - NEVER by a key written into this file or any other source
file in this repo. If disabled or unavailable, the system still works
end-to-end using only the deterministic retrieval + reasoning pipeline -
satisfying "No generic LLM-only solutions" by construction, since the LLM is
never the only source of an answer.

Two providers are supported, checked in this order:
  1. ANTHROPIC_API_KEY  -> Claude, via the official `anthropic` SDK.
  2. GROQ_API_KEY        -> Groq, via its OpenAI-compatible REST endpoint
                            (https://api.groq.com/openai/v1/chat/completions),
                            called with plain `requests` so no extra SDK is
                            required. Model defaults to `GROQ_MODEL` env var
                            or "llama-3.3-70b-versatile" if unset.

SECURITY NOTE FOR ANYONE DEPLOYING THIS: set these as real environment
variables (`export GROQ_API_KEY=...`, a platform's secrets manager, or a
local, gitignored `.env` loaded by your process manager) - never hardcode a
key here, never commit a `.env` file, and never pass a key into any
client-side/browser code (frontend/demo.html is intentionally LLM-free for
exactly this reason: anything shipped to a browser is visible to anyone who
views the page source).

When enabled, the LLM is given the ALREADY-RETRIEVED, ALREADY-STRUCTURED
Recommendation objects and asked only to (a) write a natural-language
opening summary of the assessment, and (b) smooth the phrasing of clarifying
questions. It is explicitly instructed not to add new facts, numbers, or
sources, and the prompt includes the retrieved evidence so any paraphrase
stays anchored to it. This is the "9. How hallucinations are reduced"
answer from docs/RAG_PIPELINE.md in code form.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
_GROQ_KEY = os.environ.get("GROQ_API_KEY")
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_ENABLED = bool(_ANTHROPIC_KEY or _GROQ_KEY)

SYSTEM_PROMPT = """You are a phrasing assistant for an environmental science system.
You will be given a JSON object containing an environmental assessment,
variable relationships, and evidence-backed recommendations that have
ALREADY been computed and verified. Your only job is to write a clear,
natural-language 2-4 sentence introductory summary that orients the user.

STRICT RULES:
- Do not introduce any new number, percentage, statistic, study, author, or
  citation that is not already present in the JSON you were given.
- Do not soften, remove, or override any uncertainty language (e.g. "the
  literature is contested", "not quantified") that is present in the input.
- If you are unsure whether something is already in the input, leave it out.
- Output plain text only, no markdown headers.
"""


def is_enabled() -> bool:
    return _ENABLED


def active_provider() -> Optional[str]:
    if _ANTHROPIC_KEY:
        return "anthropic"
    if _GROQ_KEY:
        return "groq"
    return None


def _summarize_via_anthropic(structured_payload: Dict[str, Any]) -> Optional[str]:
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": str(structured_payload)}],
    )
    text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return "\n".join(text_blocks).strip() or None


def _summarize_via_groq(structured_payload: Dict[str, Any]) -> Optional[str]:
    import requests  # type: ignore

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {_GROQ_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": _GROQ_MODEL,
            "max_tokens": 300,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(structured_payload)},
            ],
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return text.strip() or None


def summarize_assessment(structured_payload: Dict[str, Any]) -> Optional[str]:
    """
    Returns a short natural-language framing summary, or None if the LLM
    layer is disabled/unavailable/fails (caller falls back to the
    deterministic template string in that case - see pipeline.py).
    """
    provider = active_provider()
    if not provider:
        return None
    try:
        if provider == "anthropic":
            return _summarize_via_anthropic(structured_payload)
        if provider == "groq":
            return _summarize_via_groq(structured_payload)
    except Exception:
        # Network/key/library/rate-limit issues should never break the
        # deterministic pipeline - degrade gracefully to the template summary.
        return None
    return None
