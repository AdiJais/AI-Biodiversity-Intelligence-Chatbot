"""
Recommendation + Citation/Confidence Agent.

Turns (known variables + retrieved evidence + reasoning chains) into the
structured Recommendation objects defined in app/models/schemas.py. This
module NEVER invents a statistic: every `expected_impact` string is built
directly from a `quantitative_effect` field that a human wrote from a real
source in knowledge_base.json, or is explicitly phrased as directional/
qualitative when no safe number exists (see kb010-kb012).
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.models.schemas import Recommendation, EvidenceItem, ConfidenceLevel
from app.rag.retriever import retrieve, RetrievalResult
from app.reasoning.multi_metric_reasoner import relevant_edges

# Maps a KB document's "mechanism"/topic to a concrete recommended action.
# Kept as an explicit lookup (not LLM-generated) so the "what to do" text is
# always one of a small set of vetted, specific interventions - satisfying
# the brief's "Not acceptable: Use sustainable practices" constraint.
_ACTION_BY_KB_TOPIC = {
    "kb001": "Introduce a legume-based cover crop (e.g. cowpea, clover, vetch) in the off-season or between rows.",
    "kb002": "Maintain continuous cover cropping rather than leaving soil bare between cash-crop cycles.",
    "kb003": "Where a legume cover crop is already used, prioritize residue retention (don't remove biomass) so necromass carbon can accumulate.",
    "kb004": "Convert a portion of the monoculture area to an alley-cropping or silvopastoral agroforestry layout.",
    "kb005": "Retain or reintroduce native woody rows/strips within the cropped area rather than clearing to full monoculture.",
    "kb006": "Prioritize protecting/expanding total habitat AREA first; treat patch shape/fragmentation as a secondary, less certain lever.",
    "kb007": "Establish or widen a vegetated riparian buffer strip (aim for >=100 ft / 30 m where land allows) along adjacent waterways.",
    "kb008": "Prioritize soil-cover and erosion-control measures immediately - this region's baseline degradation risk is already elevated.",
    "kb009": "Treat soil-carbon-building interventions as climate adaptation, not just a biodiversity measure, given reinforcing land-climate feedback.",
    "kb010": "Shift to reduced/no-till with permanent soil cover and crop rotation (FAO's 3 conservation agriculture principles).",
    "kb011": "Diversify the monoculture with intercropped species or flowering field-margin strips to support pollinators and natural pest control.",
    "kb012": "Test and, if needed, amend soil pH toward the 6.0-7.5 range before investing in other soil-carbon interventions.",
}

_TIME_HORIZON_BY_KB = {
    "kb001": "Medium term (2-3 seasons to see measurable SOC change)",
    "kb002": "Medium to long term (effect size grows with duration under cover)",
    "kb003": "Medium term (multi-season, orchard/perennial systems)",
    "kb004": "Long term (multi-year tree establishment, carbon accrues annually)",
    "kb005": "Long term (semi-arid biomass accrual is slow)",
    "kb006": "Long term (landscape-scale habitat planning)",
    "kb007": "Short to medium term (buffers function within 1-2 growing seasons once vegetated)",
    "kb008": "Short term for erosion control; long term for full ecosystem recovery",
    "kb009": "Long term (climate feedback timescales)",
    "kb010": "Medium term (soil structure benefits build over 2-5 years)",
    "kb011": "Short to medium term (pollinator response can appear within 1 season)",
    "kb012": "Short term (pH amendment effects appear within one season)",
}


def _confidence_from_evidence(results: List[RetrievalResult]) -> tuple[ConfidenceLevel, str]:
    if not results:
        return ConfidenceLevel.low, "No closely matching evidence was retrieved for this specific situation."
    top = results[0]
    kb_conf = top.document.get("confidence", "medium")
    evidence_type = top.document.get("evidence_type", "")
    if "meta-analysis" in evidence_type or "systematic review" in evidence_type:
        base = ConfidenceLevel.high
    elif "regional" in evidence_type or "government" in evidence_type or "IPCC" in evidence_type or "policy" in evidence_type:
        base = ConfidenceLevel.medium
    else:
        base = ConfidenceLevel.medium

    if "low" in str(kb_conf):
        base = ConfidenceLevel.low

    explanation = (
        f"Top evidence is a {evidence_type} (source-level confidence: {kb_conf}); "
        f"retrieval similarity score {top.score:.2f}."
    )
    if top.document.get("evidence_notes"):
        explanation += " Note: " + top.document["evidence_notes"]
    return base, explanation


def generate_recommendations(
    known_variables: Dict[str, Any],
    max_recommendations: int = 3,
    raw_text: str = "",
) -> List[Recommendation]:
    variable_keys = list(known_variables.keys())

    # Build a query string from known variables for the retriever - this is
    # the "Query Understanding" -> "Knowledge Retrieval" handoff. The raw
    # message text is blended in (weighted by repetition) so a follow-up
    # question ("how do I reduce water pollution here?") with no NEW
    # extractable variable still re-ranks retrieval against what was just
    # asked, instead of only ever re-ranking against accumulated state.
    query_parts = []
    for k, v in known_variables.items():
        query_parts.append(f"{k.replace('_', ' ')}: {v}")
    state_str = "; ".join(query_parts)
    free_text = (raw_text or "").strip()
    query = ". ".join(p for p in [free_text, free_text, state_str] if p) or "general biodiversity improvement"

    results = retrieve(query, variable_keys=variable_keys, top_k=6)

    # De-duplicate to one recommendation per KB source, ranked by score.
    seen_kb_ids = set()
    recommendations: List[Recommendation] = []
    for r in results:
        kb_id = r.document["id"]
        if kb_id in seen_kb_ids:
            continue
        seen_kb_ids.add(kb_id)

        action = _ACTION_BY_KB_TOPIC.get(kb_id, "Apply the evidence-based practice described below.")
        qty = r.document.get("quantitative_effect", {})
        if qty.get("value_type") == "qualitative_inference":
            impact_str = f"Direction: {qty.get('direction', 'uncertain')}. Magnitude: {qty.get('magnitude', 'not quantified - treat as directional evidence only.')}"
        else:
            impact_str = f"{qty.get('magnitude', 'see evidence')} (direction: {qty.get('direction', 'n/a')}; {qty.get('value_type', 'reported value')})"

        conf, conf_explanation = _confidence_from_evidence([r])

        evidence_item = EvidenceItem(
            kb_id=kb_id,
            title=r.document["title"],
            source=r.document["source"],
            year=r.document.get("year"),
            url=r.document.get("url"),
            evidence_type=r.document.get("evidence_type", ""),
            finding_summary=r.document.get("finding_summary", ""),
            confidence=str(r.document.get("confidence", "medium")),
            relevance_score=round(r.score, 3),
        )

        rec = Recommendation(
            what_to_do=action,
            scientific_reasoning=r.document.get("finding_summary", ""),
            environmental_mechanism=r.document.get("mechanism", ""),
            impacted_metrics=r.document.get("variables_studied", []),
            expected_impact=impact_str,
            time_horizon=_TIME_HORIZON_BY_KB.get(kb_id, "Medium term"),
            evidence=[evidence_item],
            confidence=conf,
            confidence_explanation=conf_explanation,
        )
        recommendations.append(rec)
        if len(recommendations) >= max_recommendations:
            break

    return recommendations
