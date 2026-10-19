.PHONY: install test test-slow lint build
install:
	pip install -e ".[dev]"
test:
	pytest -q
test-slow:
	DMGUARD_SLOW=1 pytest -q
lint:
	ruff check src tests
build:
	python -m build
