.PHONY: install install-nli lint typecheck test cov check run docker-build up down chaos demo \
        bench-data bench bench-nli bench-hybrid bench-token

install:          ## Install the package with dev tools
	pip install -e ".[dev]"

install-nli:      ## Install the NLI scorer (CPU-only torch keeps the download small)
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[nli]"

lint:             ## Ruff lint
	ruff check .

typecheck:        ## mypy --strict
	mypy

test:             ## Run the test suite
	pytest

cov:              ## Tests with coverage report
	pytest --cov --cov-report=term-missing

check: lint typecheck cov  ## Everything CI runs

run:              ## Run the gateway locally with readable logs
	GROUNDWIRE_LOG_JSON=false groundwire

docker-build:     ## Build the container image
	docker build -t groundwire:dev .

up:               ## Start gateway + Prometheus
	docker compose up --build -d

down:
	docker compose down

chaos:            ## Fire 500 requests at an already-running gateway (start it with `make run`)
	python scripts/chaos_demo.py --requests 500 --concurrency 50

demo:             ## One command: start a temporary gateway, run the chaos test, stop it
	./scripts/run_demo.sh --requests 500 --concurrency 50

bench-data:       ## Download the RAGTruth benchmark into data/ragtruth
	./scripts/download_ragtruth.sh

bench: bench-data ## Lexical scorer on the full RAGTruth test set (~1 minute, no extra deps)
	python -m groundwire.evals --scorer lexical --per-task 0 --dev-per-task 200

bench-nli: bench-data  ## NLI scorer on 100 test responses per task (needs make install-nli)
	python -m groundwire.evals --scorer nli --per-task 100 --dev-per-task 50 --evidence-passages 3

bench-hybrid: bench-data  ## Numeric gate + NLI (needs make install-nli)
	python -m groundwire.evals --scorer hybrid --per-task 100 --dev-per-task 50 --evidence-passages 3

bench-token: bench-data  ## LettuceDetect token classifier, full test set, native 0.5 threshold
	python -m groundwire.evals --scorer token --per-task 0 --dev-per-task 0 \
		--threshold 0.5 --aggregation min --min-claim-tokens 1
