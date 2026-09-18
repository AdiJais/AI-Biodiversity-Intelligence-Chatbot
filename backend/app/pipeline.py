"""
Orchestrator: User Query -> Query Understanding -> Variable Extraction ->
Knowledge Retrieval -> Multi-Metric Reasoning -> Recommendation Generation
-> Evidence/Citation Layer -> Structured Response.

This is the single place that assembles the full pipeline described in
docs/ARCHITECTURE.md, so main.py's routes stay thin.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.conversation.memory import memory_store
from app.models.schemas import (
    ChatRequest,
    EnvironmentalObservation,
    StructuredResponse,
)
from app.reasoning.variable_extractor import (
    extract_from_text,
    extract_from_structured,
    merge_variables,
    missing_critical_variables,
    clarifying_questions_for,
)
from app.reasoning.multi_metric_reasoner import summarize_relationships, all_supporting_kb_ids
from app.reasoning.recommendation_engine import generate_recommendations
from app.reasoning.llm_augment import summarize_assessment, is_enabled as llm_enabled


def _template_assessment_summary(
    known_variables: Dict[str, Any],
    newly_added: List[str],
    recommendations: List,
) -> str:
    """
    Builds the reply text. Deliberately leads with a direct answer drawn from
    the top retrieved recommendation for THIS turn (not just a restatement of
    accumulated state) - a distinct question with no new extractable
    variable (e.g. "how do I reduce water pollution here?") must still get a
    distinct, relevant answer instead of an identical-looking summary line.
    """
    parts: List[str] = []

    if recommendations:
        top = recommendations[0]
        parts.append(f"{top.what_to_do}\n{top.scientific_reasoning}")
        if len(recommendations) > 1:
            parts.append(f"({len(recommendations) - 1} more option(s) available - see 'recommendations' for full evidence and confidence.)")
    elif not known_variables:
        parts.append(
            "No environmental variables detected yet. Share what you know about soil, "
            "land use, rainfall, or observed biodiversity - or ask a specific question - "
            "and I'll ground the answer in retrieved evidence rather than guessing."
        )
    else:
        parts.append(
            "No strongly matching evidence was retrieved for that specific question yet. "
            "Try adding more detail about soil, land use, rainfall, or water/pollution conditions."
        )

    if newly_added:
        desc = ", ".join(f"{k.replace('_', ' ')} = {known_variables[k]}" for k in newly_added)
        parts.append(f"Noted: {desc}.")

    return "\n\n".join(parts)


def run_pipeline(request: ChatRequest) -> StructuredResponse:
    session = memory_store.get_or_create(request.session_id)
    memory_store.record_turn(request.session_id, "user", request.message)

    prior_known = dict(session.known_variables)  # snapshot before this turn's merge

    text_vars = extract_from_text(request.message) if request.message else {}
    structured_vars = (
        extract_from_structured(request.structured) if request.structured else {}
    )
    this_turn_vars = merge_variables(text_vars, structured_vars)

    # Merge: session memory (oldest) -> newly parsed text -> explicit structured input (wins)
    merged = merge_variables(session.known_variables, this_turn_vars)
    memory_store.update_variables(request.session_id, merged)
    session = memory_store.get_or_create(request.session_id)  # refreshed

    newly_added = [
        k for k, v in this_turn_vars.items()
        if k not in prior_known or str(prior_known[k]) != str(v)
    ]

    known_keys = list(session.known_variables.keys())
    missing = missing_critical_variables(known_keys)
    all_possible_questions = clarifying_questions_for(missing)
    questions_to_ask = memory_store.filter_unasked(request.session_id, all_possible_questions)
    memory_store.record_asked_questions(request.session_id, questions_to_ask)

    relationships = summarize_relationships(known_keys)
    # Blend this turn's raw message into retrieval so a direct question -
    # not just accumulated state - can steer which evidence surfaces first.
    recommendations = (
        generate_recommendations(session.known_variables, raw_text=request.message)
        if (known_keys or request.message)
        else []
    )

    assessment_summary = _template_assessment_summary(session.known_variables, newly_added, recommendations)
    if questions_to_ask:
        assessment_summary += "\n\n" + "\n".join(f"- {q}" for q in questions_to_ask)

    if llm_enabled():
        llm_payload = {
            "known_variables": session.known_variables,
            "newly_added_this_turn": newly_added,
            "missing_variables": missing,
            "relationships": relationships,
            "recommendations": [r.model_dump() for r in recommendations],
        }
        llm_summary = summarize_assessment(llm_payload)
        if llm_summary:
            assessment_summary = llm_summary

    response = StructuredResponse(
        environmental_assessment=assessment_summary,
        key_variables=session.known_variables,
        variable_relationships=relationships,
        recommendations=recommendations,
        clarifying_questions=questions_to_ask,
        missing_variables=missing,
        session_id=request.session_id,
    )
    memory_store.record_turn(request.session_id, "assistant", assessment_summary)
    return response
