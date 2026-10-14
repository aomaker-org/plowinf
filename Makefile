# Path: plowinf/Makefile
# Makefile for plowinf workspace

SHELL := /bin/bash
PYTHON := uv run python
PYTEST := uv run pytest

.PHONY: help venv install test test-cov lint status backends benchmark daemon view watch products sync clean

help:
	@echo "plowinf Development Targets:"
	@echo "  make venv       - Create .venv & install package in editable mode"
	@echo "  make test       - Run test suite via uv run pytest"
	@echo "  make test-cov   - Run test suite with coverage report"
	@echo "  make status     - Check queue status and backend availability"
	@echo "  make backends   - Audit local inference backends"
	@echo "  make benchmark  - Run inference benchmark against active backend"
	@echo "  make daemon     - Run background worker daemon"
	@echo "  make view       - View generated artifacts and executions"
	@echo "  make watch      - Stream live completed tasks and audit logs"
	@echo "  make products   - List all generated work product markdown files"
	@echo "  make sync       - Run turn-sync handover audit (Directive 08)"
	@echo "  make clean      - Clean temporary artifacts and cache"

venv:
	@echo "[*] Creating virtual environment with uv..."
	uv venv
	@echo "[*] Installing plowinf in editable mode..."
	uv pip install -e ".[dev]"
	@echo "[+] Virtual environment ready."

install: venv

test:
	@echo "[*] Running plowinf test suite..."
	$(PYTEST) -v tests/

test-cov:
	$(PYTEST) --cov=plowinf tests/

status:
	uv run plowinf status

backends:
	uv run plowinf backends

benchmark:
	uv run plowinf benchmark

daemon:
	uv run plowinf daemon --interval 2.0

view:
	@./tools/plowinf_viewer.sh summary

watch:
	@./tools/plowinf_viewer.sh follow

products:
	@./tools/plowinf_viewer.sh products

sync: test
	@echo "=== [Directive 08] plowinf Turn-Sync Audit ==="
	@git status
	@echo "[+] Turn-sync audit completed cleanly."

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# end of file: plowinf/Makefile
