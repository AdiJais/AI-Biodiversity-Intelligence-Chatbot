from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["knowledge_base_size"] >= 10


def test_chat_missing_data_asks_clarifying_questions():
    r = client.post("/chat", json={"session_id": "test-session-1", "message": "Biodiversity is declining on my land."})
    assert r.status_code == 200
    body = r.json()
    assert len(body["clarifying_questions"]) > 0
    assert len(body["missing_variables"]) > 0


def test_chat_remembers_variables_across_turns():
    session_id = "test-session-2"
    r1 = client.post("/chat", json={"session_id": session_id, "message": "It's a monoculture wheat farm with 0.3% soil organic carbon."})
    assert r1.status_code == 200
    r2 = client.post("/chat", json={"session_id": session_id, "message": "Rainfall here is low."})
    assert r2.status_code == 200
    body2 = r2.json()
    # land_use and soc from turn 1 should still be present in turn 2's key_variables
    assert body2["key_variables"].get("land_use") == "monoculture"
    assert body2["key_variables"].get("soil_organic_carbon") == 0.3
    assert body2["key_variables"].get("rainfall") == "low"
    # all 3 critical variables now known -> no more missing critical variables
    assert body2["missing_variables"] == []
    assert len(body2["recommendations"]) > 0


def test_chat_follow_up_question_gets_a_distinct_relevant_reply_not_a_repeated_state_dump():
    """
    Regression test for a reported bug: a distinct follow-up question with no
    NEW extractable variable (e.g. asking 'how do I reduce water pollution
    here?' right after describing pollution) must produce a reply that
    actually answers the question via retrieval, not an identical-looking
    'current known conditions' restatement every time.
    """
    session_id = "test-session-bug-repro"
    r1 = client.post("/chat", json={
        "session_id": session_id,
        "message": "There's agricultural runoff and pollution getting into the stream next to my field, and water availability for wildlife seems low.",
    })
    assert r1.status_code == 200
    body1 = r1.json()
    assert len(body1["recommendations"]) > 0
    assert "riparian" in body1["environmental_assessment"].lower() or "buffer" in body1["environmental_assessment"].lower()

    r2 = client.post("/chat", json={"session_id": session_id, "message": "how do I reduce water pollution on my farm"})
    assert r2.status_code == 200
    body2 = r2.json()
    # Must still surface a concrete, evidence-backed answer (not an empty rec list)
    assert len(body2["recommendations"]) > 0
    assert body2["recommendations"][0]["evidence"][0]["kb_id"] == "kb007"
    # The reply text must be a real answer, not just "Current known conditions: ..."
    assert not body2["environmental_assessment"].lower().startswith("current known conditions")


def test_assess_structured_endpoint():
    r = client.post(
        "/assess",
        json={
            "session_id": "test-session-3",
            "observation": {
                "soil_organic_carbon_pct": 0.3,
                "rainfall": "low",
                "land_use": "monoculture",
                "crop_type": "wheat",
                "region": "semi-arid",
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) > 0
    assert any("agroforestry" in rec["what_to_do"].lower() or "cover crop" in rec["what_to_do"].lower() for rec in body["recommendations"])
