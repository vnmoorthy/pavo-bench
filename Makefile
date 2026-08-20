# PAVO-Bench — common dev tasks
PYTHON  ?= python3
PIP     ?= $(PYTHON) -m pip
RUFF    ?= $(PYTHON) -m ruff
LINT_PATHS = pavo_bench/__init__.py pavo_bench/loader.py pavo_bench/routers.py \
	pavo_bench/state.py pavo_bench/evaluate.py tests/ scripts/render_figures.py

.PHONY: help install dev test lint format figures clean repro

help:                       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN{FS=":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:                    ## Install package + runtime deps
	$(PIP) install -e .

dev:                        ## Install package + dev deps
	$(PIP) install -e ".[dev,full]"

test:                       ## Run the package smoke tests
	$(PYTHON) -m pytest -q tests/

lint:                       ## Lint with ruff
	$(RUFF) check $(LINT_PATHS)

format:                     ## Auto-format with ruff
	$(RUFF) format $(LINT_PATHS)

figures:                    ## Re-render figures from the committed JSONs
	$(PYTHON) scripts/render_figures.py

repro:                      ## Run the full GPU experiment suite
	bash experiments/setup.sh
	$(PYTHON) experiments/run_all_experiments.py --skip-upload

clean:                      ## Remove build/cache artifacts (no data files touched)
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ __pycache__/
	find . -name __pycache__ -prune -exec rm -rf {} +
