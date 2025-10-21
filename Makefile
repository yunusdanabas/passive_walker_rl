# Makefile (optional)
.PHONY: fmt lint test smoke demo
fmt:
	black passive_walker/ tests/ tools/ scripts/
lint:
	ruff check passive_walker/ tests/ tools/ scripts/
test:
	pytest -q
smoke:
	walker-demo --no-gui --seconds 5
demo:
	walker-demo --seconds 10 --save-rgb-array --gif

