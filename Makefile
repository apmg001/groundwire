.PHONY: install install-nli lint typecheck test cov check run docker-build up down chaos demo \
        bench-data bench bench-nli bench-hybrid bench-token bench-cache-data bench-cache \
        gate gate-full gate-calibrate bench-cascade bench-cascade-token \
        agent agent-risky agent-seq agent-risky-seq agent-liar \
        agent-approval agent-approval-manual approvals approve reject \
        install-load loadtest-load loadtest-stress loadtest-spike loadtest-chaos loadtest-soak \
        demo-live

# Reviewer token for the approval API (make run and the approval targets share it).
# Development default only: set your own secret in any shared or deployed setup.
GROUNDWIRE_APPROVER_TOKEN ?= dev-approver-token-change-me
export GROUNDWIRE_APPROVER_TOKEN

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

agent-liar:       ## No model needed: an agent that misreports its refund; the answer check flags it
	python scripts/agent_demo.py --model agent-demo-liar --scenario safe

agent-approval:   ## No model needed: a $1,200 refund waits for approval; the demo plays the manager
	python scripts/agent_demo.py --model agent-demo-approval --scenario approval --auto-approve

agent-approval-manual: ## Same, but YOU approve it from another terminal: make approve ID=apr_...
	python scripts/agent_demo.py --model agent-demo-approval --scenario approval --wait-approval

approvals:        ## List tool calls waiting for human approval
	python scripts/approvals.py list

approve:          ## Approve a held call: make approve ID=apr_... [NOTE="..."]
	python scripts/approvals.py approve $(ID) $(if $(NOTE),--note "$(NOTE)")

reject:           ## Reject a held call: make reject ID=apr_... [NOTE="..."]
	python scripts/approvals.py reject $(ID) $(if $(NOTE),--note "$(NOTE)")

# --------------------------------------------------------------------------- #
# Load tests: each starts two fake model servers and a gateway, runs Locust,
# and writes results/loadtest/<time>-<profile>/report.md. Stop anything on
# ports 8020, 9101 and 9102 first.
# --------------------------------------------------------------------------- #

install-load:     ## Install Locust and psutil for the load tests
	pip install -e ".[load]"

loadtest-load:    ## 100 users for 60 s, plus a baseline straight to the fake model (overhead)
	python -m loadtest.run load --users 100 --duration 60 --baseline

loadtest-stress:  ## 25 -> 600 users in 3 minutes: where is the limit, and does shedding hold it?
	python -m loadtest.run stress

loadtest-spike:   ## 20 -> 300 -> 20 users: survive the burst and recover
	python -m loadtest.run spike

loadtest-chaos:   ## 100 users while the primary model fails, then hangs: failover under load
	python -m loadtest.run chaos --users 100 --duration 120

loadtest-soak:    ## 100 users for 10 minutes: memory must stay flat
	python -m loadtest.run soak --users 100 --duration 600

# --------------------------------------------------------------------------- #
# Live demo: the public-demo config with the dashboard, on http://localhost:7860
# Set GROQ_API_KEY for real answers; without it a backup model answers.
# --------------------------------------------------------------------------- #

demo-live:        ## Run the public demo locally: open http://localhost:7860
	GROUNDWIRE_CONFIG_PATH=config/demo.yaml GROUNDWIRE_PORT=7860 GROUNDWIRE_LOG_JSON=false \
	GROUNDWIRE_APPROVER_TOKEN=public-demo-reviewer-token groundwire
