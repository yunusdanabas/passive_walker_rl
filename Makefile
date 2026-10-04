# Makefile (optional)
.PHONY: install fmt lint test smoke demo
install:
	pip install -r requirements-dev.txt && pip install -e ".[all]"
fmt:
	black passive_walker/ tests/ tools/ scripts/
lint:
	ruff check passive_walker/ tests/ tools/ scripts/
test:
	pytest -q
smoke:
	walker-demo --no-gui --seconds 5
demo:
	walker-demo --seconds 10

