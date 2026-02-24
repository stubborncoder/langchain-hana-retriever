.PHONY: install test lint format type-check all clean build publish

install:
	uv pip install -e ".[dev]"

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/test_utils.py tests/test_bm25.py tests/test_hybrid.py -v

test-integration:
	uv run pytest tests/test_integration.py -v

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	uv run mypy langchain_hana_retriever/

all: lint type-check test

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

build:
	uv build

publish: build
	uv publish
