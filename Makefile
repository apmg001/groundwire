.PHONY: install install-nli lint typecheck test cov check run docker-build up down chaos demo \
        bench-data bench bench-nli bench-hybrid bench-token bench-cache-data bench-cache \
        gate gate-full gate-calibrate bench-cascade bench-cascade-token \
        agent agent-risky agent-seq agent-risky-seq

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

bench-cache-data: ## Download Quora Question Pairs (GLUE) into data/qqp (~37 MB)
	./scripts/download_qqp.sh
	pip install -q -e ".[bench]"

bench-cache: bench-cache-data  ## Semantic-cache hit/false-hit rates (needs make install-nli)
	python -m groundwire.evals.cachebench --tune-pairs 20000 --test-pairs 0

SUITE ?= fast

gate: bench-data  ## Quality gate CI runs on every push (lexical, ~5 s, no ML deps)
	python -m groundwire.evals.gate --suite fast

gate-full: bench-data bench-cache-data  ## Model-based quality gates (needs make install-nli)
	python -m groundwire.evals.gate --suite full

gate-calibrate: bench-data $(if $(filter full,$(SUITE)),bench-cache-data)  ## Print suggested bounds for SUITE=fast|full (review, then commit)
	python -m groundwire.evals.gate --suite $(SUITE) --calibrate

bench-cascade: bench-data  ## Cheap-first routing simulation, lexical check (no ML deps, ~5 s)
	python -m groundwire.evals.cascade --cheap llama-2-7b-chat
	python -m groundwire.evals.cascade --cheap gpt-3.5-turbo-0613

bench-cascade-token: bench-data  ## Same with LettuceDetect as the check (needs make install-nli, ~15 min)
	python -m groundwire.evals.cascade --cheap llama-2-7b-chat --scorer token
	python -m groundwire.evals.cascade --cheap gpt-3.5-turbo-0613 --scorer token

agent:            ## Real tool-using agent via the gateway (needs `make run` + Ollama with qwen2.5:3b)
	python scripts/agent_demo.py --model ollama --scenario safe

agent-risky:      ## Same agent, a refund above the policy cap: watch the gateway block it
	python scripts/agent_demo.py --model ollama --scenario risky

agent-seq:        ## Safe scenario, one tool call per step (look up first, then refund)
	python scripts/agent_demo.py --model ollama --scenario safe --sequential

agent-risky-seq:  ## Risky scenario, one tool call per step
	python scripts/agent_demo.py --model ollama --scenario risky --sequential
