from app.reasoning.recommendation_engine import generate_recommendations


def test_recommendations_are_grounded_in_evidence():
    known = {"soil_organic_carbon": 0.3, "rainfall": "low", "land_use": "monoculture", "crop_type": "wheat"}
    recs = generate_recommendations(known)
    assert len(recs) > 0
    for rec in recs:
        assert len(rec.evidence) >= 1
        for ev in rec.evidence:
            # every recommendation must cite a real, non-empty source string
            assert ev.source
            assert ev.title
        assert rec.time_horizon
        assert rec.confidence in {"High", "Medium", "Low"}


def test_qualitative_kb_entries_never_produce_a_fabricated_percent_in_expected_impact():
    known = {"soil_ph": 9.5}  # should surface kb012 (qualitative, no invented number)
    recs = generate_recommendations(known)
    assert len(recs) > 0
    for rec in recs:
        if any(ev.kb_id in {"kb010", "kb011", "kb012"} for ev in rec.evidence):
            assert "not quantified" in rec.expected_impact or "qualitative" in rec.expected_impact.lower() or "%" not in rec.expected_impact
