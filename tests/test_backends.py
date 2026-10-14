# Path: plowinf/tests/test_backends.py
"""Unit tests for local SLM backend adapters and router in plowinf."""

from unittest.mock import MagicMock, patch
import pytest

from plowinf.backends import (
    BackendRouter,
    MockBackend,
    OllamaBackend,
    OpenAICompatBackend,
)


def test_mock_backend_availability():
    b = MockBackend()
    assert b.is_available() is True
    assert b.name == "mock"


def test_mock_backend_generation():
    b = MockBackend(prefix="[TEST-MOCK]")
    res = b.generate(prompt="def add(a, b): return a + b")
    assert "[TEST-MOCK]" in res
    assert "Code Analysis Complete" in res
    assert "Deterministic verification PASSED" in res


def test_router_selects_mock_when_others_unavailable():
    mock_b = MockBackend()
    ollama_b = OllamaBackend(base_url="http://127.0.0.1:99999")
    router = BackendRouter(backends=[ollama_b, mock_b])

    active = router.get_active_backend()
    assert active.name == "mock"

    res = router.generate("test code")
    assert "Code Analysis Complete" in res


def test_ollama_backend_unreachable():
    b = OllamaBackend(base_url="http://127.0.0.1:99999", timeout=0.1)
    assert b.is_available() is False


def test_openai_compat_backend_unreachable():
    b = OpenAICompatBackend(base_url="http://127.0.0.1:99999/v1", timeout=0.1)
    assert b.is_available() is False


@patch("requests.post")
def test_openai_compat_successful_generate(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Generated test code"}}]
    }
    mock_post.return_value = mock_resp

    b = OpenAICompatBackend(base_url="http://127.0.0.1:8000/v1")
    output = b.generate(prompt="write test")
    assert output == "Generated test code"


@patch("requests.get")
def test_ollama_list_models(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "qwen2:1.5b"}, {"name": "llama3:8b"}]
    }
    mock_get.return_value = mock_resp

    b = OllamaBackend()
    models = b.list_models()
    assert "qwen2:1.5b" in models
    assert "llama3:8b" in models


@patch("requests.post")
def test_ollama_successful_generate_with_options(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "def quicksort(): pass"}
    mock_post.return_value = mock_resp

    b = OllamaBackend()
    output = b.generate(prompt="sort", temperature=0.5, max_tokens=256)
    assert output == "def quicksort(): pass"
    assert mock_post.called
    payload = mock_post.call_args[1]["json"]
    assert payload["options"]["temperature"] == 0.5
    assert payload["options"]["num_predict"] == 256


# end of file: plowinf/tests/test_backends.py
