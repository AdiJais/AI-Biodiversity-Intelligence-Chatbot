from app.rag.retriever import retrieve


def test_retrieve_returns_results_for_cover_crop_query():
    results = retrieve("legume cover crop soil organic carbon semi-arid monoculture wheat", top_k=3)
    assert len(results) > 0
    top_ids = [r.document["id"] for r in results]
    # kb001/kb002/kb003 are the cover-crop/SOC sources; at least one should surface
    assert any(i in {"kb001", "kb002", "kb003"} for i in top_ids)


def test_retrieve_metadata_filter_narrows_to_relevant_variable():
    results = retrieve("water quality nitrogen removal", variable_keys=["water_availability", "pollution"], top_k=5)
    assert len(results) > 0
    assert results[0].document["id"] == "kb007"


def test_retrieve_never_returns_empty_for_nonempty_corpus():
    results = retrieve("asdkjfhaskjdfh nonsense query xyz", top_k=3)
    assert len(results) >= 1  # guarantees at least the closest match, never a silent empty set
