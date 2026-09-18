"""
Query Understanding + Environmental Variable Extraction Agent.

Converts free text and/or a structured EnvironmentalObservation into a
normalized dict of {variable_key: value} using the canonical variable keys
declared in data/variable_interactions.json, plus a list of which
"critical" variables are still missing. This is deliberately rule/regex
based rather than an LLM call: extraction correctness is easy to test and
audit this way, and it keeps the pipeline working with zero API calls.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.knowledge.loader import load_interaction_graph
from app.models.schemas import EnvironmentalObservation

# Variables we consider "critical" to ask about if entirely absent from the
# conversation. Mirrors the brief's example: "Can you provide soil organic
# carbon %, rainfall pattern, and land use type?"
CRITICAL_VARIABLES = ["soil_organic_carbon", "rainfall", "land_use"]

_ALL_VARIABLE_KEYS = [v["key"] for v in load_interaction_graph()["variables"]]

# --- Lightweight regex/keyword extraction from free text -------------------

# Matches either order: "0.3% soil organic carbon" or "soil organic carbon of 0.3%"
_NUMERIC_SOC_AFTER = re.compile(
    r"(soil\s+organic\s+carbon|SOC|organic\s+carbon)[^0-9%]{0,15}([0-9]+(?:\.[0-9]+)?)\s*%"
)
_NUMERIC_SOC_BEFORE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*%[^a-zA-Z]{0,10}(soil\s+organic\s+carbon|SOC|organic\s+carbon)", re.I
)
_NUMERIC_PH = re.compile(r"\b(?:soil\s+)?ph\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\b", re.I)
_NUMERIC_TEMP = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:°\s*c|deg(?:rees)?\s*c|celsius)\b", re.I)

_RAINFALL_WORDS = {
    "very_low": ["very low rainfall", "rainfall is very low", "almost no rain", "drought"],
    "low": ["low rainfall", "rainfall is low", "rainfall here is low", "semi-arid", "semiarid", "dry region", "arid"],
    "moderate": ["moderate rainfall", "rainfall is moderate", "average rainfall"],
    "high": ["high rainfall", "rainfall is high", "wet region", "monsoon"],
    "very_high": ["very high rainfall", "rainfall is very high", "extremely wet"],
}
# Fallback: "rainfall ... <level>" within a short window, catches phrasing
# variants not enumerated above (e.g. "rainfall around here is quite low").
_RAINFALL_PROXIMITY = re.compile(
    r"rainfall\b(?:[^.]{0,25})\b(very low|very high|low|moderate|high)\b", re.I
)
_RAINFALL_LEVEL_MAP = {
    "very low": "very_low", "low": "low", "moderate": "moderate",
    "high": "high", "very high": "very_high",
}

_LAND_USE_WORDS = {
    "monoculture": ["monoculture", "single crop", "mono-crop", "mono crop"],
    "intercropping": ["intercropping", "inter-cropping", "mixed cropping"],
    "agroforestry": ["agroforestry", "agro-forestry", "alley cropping", "silvopasture"],
    "pasture_grazing": ["pasture", "grazing", "rangeland", "livestock grazing"],
    "forest": ["forest", "woodland"],
    "urban": ["urban", "built-up"],
    "wetland": ["wetland", "marsh", "swamp"],
}

_DEFORESTATION_WORDS = ["deforestation", "clear-cut", "clearcut", "cleared forest", "logging"]
_POLLUTION_WORDS = ["pollution", "pesticide runoff", "fertilizer runoff", "agrochemical", "contamina"]
_LOW_MOISTURE_WORDS = ["dry soil", "low soil moisture", "parched", "drought-stressed soil"]


def _match_keyword_set(text: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    text_l = text.lower()
    for key, phrases in mapping.items():
        for phrase in phrases:
            if phrase in text_l:
                return key
    return None


def extract_from_text(text: str) -> Dict[str, Any]:
    found: Dict[str, Any] = {}

    m = _NUMERIC_SOC_AFTER.search(text)
    if m:
        found["soil_organic_carbon"] = float(m.group(2))
    else:
        m = _NUMERIC_SOC_BEFORE.search(text)
        if m:
            found["soil_organic_carbon"] = float(m.group(1))

    m = _NUMERIC_PH.search(text)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 14:
            found["soil_ph"] = val

    m = _NUMERIC_TEMP.search(text)
    if m:
        found["temperature"] = float(m.group(1))

    rainfall = _match_keyword_set(text, _RAINFALL_WORDS)
    if not rainfall:
        m = _RAINFALL_PROXIMITY.search(text)
        if m:
            rainfall = _RAINFALL_LEVEL_MAP.get(m.group(1).lower())
    if rainfall:
        found["rainfall"] = rainfall

    land_use = _match_keyword_set(text, _LAND_USE_WORDS)
    if land_use:
        found["land_use"] = land_use

    text_l = text.lower()
    if any(w in text_l for w in _DEFORESTATION_WORDS):
        found["deforestation"] = "present"
    if any(w in text_l for w in _POLLUTION_WORDS):
        found["pollution"] = "present"
    if any(w in text_l for w in _LOW_MOISTURE_WORDS):
        found["soil_moisture"] = "low"

    # crop type: naive "wheat", "maize", "rice", "cotton" etc. keyword catch
    for crop in ["wheat", "maize", "corn", "rice", "cotton", "soybean", "sorghum", "millet", "barley"]:
        if crop in text_l:
            found["crop_type"] = crop
            break

    return found


def extract_from_structured(obs: EnvironmentalObservation) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    if obs.soil_ph is not None:
        found["soil_ph"] = obs.soil_ph
    if obs.soil_organic_carbon_pct is not None:
        found["soil_organic_carbon"] = obs.soil_organic_carbon_pct
    if obs.soil_moisture is not None:
        found["soil_moisture"] = obs.soil_moisture
    if obs.land_use is not None:
        found["land_use"] = obs.land_use.value
    if obs.crop_type is not None:
        found["crop_type"] = obs.crop_type
    if obs.rainfall is not None:
        found["rainfall"] = obs.rainfall.value
    if obs.temperature_c is not None:
        found["temperature"] = obs.temperature_c
    if obs.water_availability is not None:
        found["water_availability"] = obs.water_availability
    if obs.pollution_level is not None:
        found["pollution"] = obs.pollution_level
    if obs.deforestation_present is not None:
        found["deforestation"] = "present" if obs.deforestation_present else "absent"
    if obs.species_richness_observed is not None:
        found["species_richness"] = obs.species_richness_observed
    return found


def merge_variables(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Later dicts override earlier ones (structured input wins over parsed text)."""
    merged: Dict[str, Any] = {}
    for d in dicts:
        merged.update({k: v for k, v in d.items() if v is not None})
    return merged


def missing_critical_variables(known_keys: List[str]) -> List[str]:
    return [v for v in CRITICAL_VARIABLES if v not in known_keys]


def clarifying_questions_for(missing: List[str]) -> List[str]:
    prompts = {
        "soil_organic_carbon": "What is the soil organic carbon percentage (or your best estimate) for this land?",
        "rainfall": "How would you describe the rainfall pattern here - very low, low, moderate, high, or very high?",
        "land_use": "What is the current land use / cropping system - e.g. monoculture, intercropping, agroforestry, pasture, or forest?",
        "soil_ph": "Do you know the soil pH?",
        "soil_moisture": "How would you describe current soil moisture - dry, moderate, or consistently moist?",
        "temperature": "What's the typical temperature range for this site or season?",
        "water_availability": "Is water availability a known constraint here (e.g. seasonal, scarce, adequate)?",
        "pollution": "Are there any known pollution sources nearby (agrochemical runoff, industrial discharge)?",
        "deforestation": "Has there been recent deforestation or vegetation clearing on or near this land?",
    }
    return [prompts[k] for k in missing if k in prompts]
