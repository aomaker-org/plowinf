# Path: plowinf/tests/test_cli.py
"""Unit tests for the plowinf command-line interface."""

from unittest.mock import patch
import pytest

from plowinf.cli import build_backend, main
from plowinf.backends import MockBackend, OllamaBackend


def test_build_backend_mock():
    b = build_backend("mock")
    assert isinstance(b, MockBackend)
    assert b.name == "mock"


def test_build_backend_ollama():
    b = build_backend("ollama", model="qwen2:1.5b")
    assert isinstance(b, OllamaBackend)
    assert b.name == "ollama"
    assert b.raw_model_spec == "qwen2:1.5b"


def test_cli_backends_command(capsys):
    with patch("sys.argv", ["plowinf", "backends"]):
        main()
    out = capsys.readouterr().out
    assert "SLM Backend Availability Audit" in out
    assert "mock" in out
    assert "ollama" in out


def test_cli_status_command(capsys):
    with patch("sys.argv", ["plowinf", "status"]):
        main()
    out = capsys.readouterr().out
    assert "File Queue Status" in out
    assert "incoming" in out
    assert "Active Backend" in out


def test_cli_benchmark_command(capsys):
    with patch("sys.argv", ["plowinf", "benchmark", "--backend", "mock", "--runs", "2"]):
        main()
    out = capsys.readouterr().out
    assert "AI Inference Benchmark: mock" in out
    assert "Run #1" in out
    assert "Run #2" in out
    assert "Average Latency" in out


# end of file: plowinf/tests/test_cli.py
