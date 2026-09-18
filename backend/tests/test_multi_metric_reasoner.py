from app.reasoning.multi_metric_reasoner import build_chains, summarize_relationships


def test_build_chains_connects_rainfall_to_species_richness():
    chains = build_chains(["rainfall"])
    endpoints = set()
    for chain in chains:
        if chain:
            endpoints.add(chain[-1]["to"])
    assert "species_richness" in endpoints  # rainfall -> soil_moisture -> species_richness


def test_summarize_relationships_uses_at_least_three_variables_when_given():
    known = ["soil_organic_carbon", "rainfall", "land_use"]
    lines = summarize_relationships(known)
    assert len(lines) > 0
    joined = " ".join(lines)
    # multi-metric reasoning requirement: connects >= 3 variables together somewhere in the output
    touched_vars = set()
    for v in ["soil_organic_carbon", "rainfall", "land_use", "soil_moisture", "species_richness", "habitat_diversity"]:
        if v in joined:
            touched_vars.add(v)
    assert len(touched_vars) >= 3
