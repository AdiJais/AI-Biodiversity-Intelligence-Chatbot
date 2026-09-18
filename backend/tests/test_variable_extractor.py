from app.reasoning.variable_extractor import (
    extract_from_text,
    missing_critical_variables,
    clarifying_questions_for,
    merge_variables,
)


def test_extract_from_text_finds_soc_and_land_use_and_rainfall():
    text = "My field has 0.3% soil organic carbon, it's a monoculture wheat farm, and rainfall is low."
    found = extract_from_text(text)
    assert found["soil_organic_carbon"] == 0.3
    assert found["land_use"] == "monoculture"
    assert found["rainfall"] == "low"
    assert found["crop_type"] == "wheat"


def test_missing_critical_variables_detects_gaps():
    missing = missing_critical_variables(["soil_organic_carbon"])
    assert "rainfall" in missing
    assert "land_use" in missing
    assert "soil_organic_carbon" not in missing


def test_clarifying_questions_only_for_missing():
    qs = clarifying_questions_for(["rainfall", "land_use"])
    assert len(qs) == 2
    assert any("rainfall" in q.lower() for q in qs)


def test_merge_variables_structured_overrides_text():
    text_vars = {"land_use": "monoculture"}
    structured_vars = {"land_use": "agroforestry"}
    merged = merge_variables(text_vars, structured_vars)
    assert merged["land_use"] == "agroforestry"
