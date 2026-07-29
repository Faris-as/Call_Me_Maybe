PYTHON := python3

TEMP := __pycache__ .mypy_cache
VENV := .venv
SRC = src

install:
	uv sync

run:
	$(PYTHON) call_me_maybe.py

clean:
	@for dir in $(TEMP); do \
			find . -type d -name "$$dir" -exec rm -rf {} +; \
		done
		find . -name "*.pyc" -delete

lint:
	$(PYTHON) -m flake8 $(SRC)
	mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs $(SRC)

lint-strict:
	flake8 $(SRC)
	mypy $(SRC) --strict

.PHONY: install run clean debug lint lint-strict test
