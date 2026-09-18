"""
Pydantic schemas for the Darukaa Biodiversity Intelligence API.

These define the two supported input modes (free text and structured JSON),
the internal representation of an "environmental observation", and the
structured response contract described in the hackathon brief's Output
Quality section.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class LandUseType(str, Enum):
    monoculture = "monoculture"
    intercropping = "intercropping"
    agroforestry = "agroforestry"
    pasture_grazing = "pasture_grazing"
    forest = "forest"
    urban = "urban"
    wetland = "wetland"
    other = "other"


class RainfallLevel(str, Enum):
    very_low = "very_low"
    low = "low"
    moderate = "moderate"
    high = "high"
    very_high = "very_high"


class ConfidenceLevel(str, Enum):
    high = "High"
    medium = "Medium"
    low = "Low"


class GeoContext(BaseModel):
    """Optional spatial context (bonus requirement in the brief)."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region_name: Optional[str] = Field(
        default=None, description="Free-text region, e.g. 'semi-arid Deccan plateau'"
    )


class EnvironmentalObservation(BaseModel):
    """
    Structured JSON input schema (mandatory bonus requirement: 'Structured
    input (JSON or similar)'). All fields are optional because the system
    must be able to work with partial data and ask clarifying questions for
    what's missing (Conversational Intelligence requirement).
    """
    soil_ph: Optional[float] = Field(default=None, ge=0, le=14)
    soil_organic_carbon_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="Soil organic carbon, percent by mass"
    )
    soil_moisture: Optional[str] = Field(
        default=None, description="Qualitative (e.g. 'low') or numeric % as string"
    )
    land_use: Optional[LandUseType] = None
    crop_type: Optional[str] = None
    rainfall: Optional[RainfallLevel] = None
    temperature_c: Optional[float] = None
    water_availability: Optional[str] = None
    pollution_level: Optional[str] = None
    deforestation_present: Optional[bool] = None
    species_richness_observed: Optional[str] = Field(
        default=None, description="Qualitative note on observed species diversity, if known"
    )
    geo: Optional[GeoContext] = None
    region: Optional[str] = None

    @field_validator("soil_organic_carbon_pct")
    @classmethod
    def _sanity_check_soc(cls, v):
        if v is not None and v > 20:
            # SOC % this high is extremely rare in mineral soils (peat excluded);
            # we don't reject it, but downstream code should flag it for confirmation.
            pass
        return v

    def populated_fields(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None and k not in ("geo",)}


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated id to maintain multi-turn memory")
    message: str
    structured: Optional[EnvironmentalObservation] = None


class EvidenceItem(BaseModel):
    kb_id: str
    title: str
    source: str
    year: Optional[int]
    url: Optional[str]
    evidence_type: str
    finding_summary: str
    confidence: str
    relevance_score: float


class Recommendation(BaseModel):
    what_to_do: str
    scientific_reasoning: str
    environmental_mechanism: str
    impacted_metrics: List[str]
    expected_impact: str
    time_horizon: str
    evidence: List[EvidenceItem]
    confidence: ConfidenceLevel
    confidence_explanation: str


class StructuredResponse(BaseModel):
    environmental_assessment: str
    key_variables: Dict[str, Any]
    variable_relationships: List[str]
    recommendations: List[Recommendation]
    clarifying_questions: List[str] = Field(default_factory=list)
    missing_variables: List[str] = Field(default_factory=list)
    session_id: str
