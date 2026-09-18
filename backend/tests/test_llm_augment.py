"""
Tests the optional LLM layer's *wiring*, not any real provider - no network
calls are made. This environment has no outbound access to api.groq.com or
api.anthropic.com, and real credentials should never live in a test file
anyway, so both provider calls are mocked at the HTTP/SDK boundary.
"""
import importlib
from unittest.mock import patch, MagicMock


def _reload_with_env(monkeypatch, **env):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from app.reasoning import llm_augment
    return importlib.reload(llm_augment)


def test_disabled_when_no_keys_present(monkeypatch):
    mod = _reload_with_env(monkeypatch)
    assert mod.is_enabled() is False
    assert mod.active_provider() is None
    assert mod.summarize_assessment({"foo": "bar"}) is None


def test_groq_is_used_when_only_groq_key_set(monkeypatch):
    mod = _reload_with_env(monkeypatch, GROQ_API_KEY="fake-test-key-not-real")
    assert mod.is_enabled() is True
    assert mod.active_provider() == "groq"

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "A grounded, evidence-based summary."}}]}

    with patch("requests.post", return_value=fake_response) as mock_post:
        result = mod.summarize_assessment({"known_variables": {"rainfall": "low"}})

    assert result == "A grounded, evidence-based summary."
    called_url = mock_post.call_args[0][0]
    assert called_url == "https://api.groq.com/openai/v1/chat/completions"
    called_kwargs = mock_post.call_args[1]
    assert called_kwargs["headers"]["Authorization"] == "Bearer fake-test-key-not-real"


def test_anthropic_key_takes_priority_over_groq(monkeypatch):
    mod = _reload_with_env(monkeypatch, ANTHROPIC_API_KEY="fake-anthropic-key", GROQ_API_KEY="fake-groq-key")
    assert mod.active_provider() == "anthropic"


def test_groq_failure_degrades_to_none_not_an_exception(monkeypatch):
    mod = _reload_with_env(monkeypatch, GROQ_API_KEY="fake-test-key-not-real")
    with patch("requests.post", side_effect=ConnectionError("no network in this sandbox")):
        result = mod.summarize_assessment({"known_variables": {}})
    assert result is None  # caller (pipeline.py) falls back to the deterministic template
