.PHONY: install lint typecheck test test-pg check run up down

# Local Postgres for the storage contract tests (any Postgres 14+ with pgvector).
MONEY_TEST_DATABASE_URL ?= postgresql://postgres@127.0.0.1:5432/money_test

install:   ## Install the app with dev tools (and the gateway, for end-to-end tests)
	pip install -e ../.. -e ".[dev]"

lint:      ## Ruff
	ruff check .

typecheck: ## mypy --strict
	mypy

test:      ## All tests; Postgres tests are skipped without MONEY_TEST_DATABASE_URL
	pytest --cov --cov-report=term-missing

test-pg:   ## All tests, including the storage contract on Postgres
	MONEY_TEST_DATABASE_URL=$(MONEY_TEST_DATABASE_URL) pytest --cov --cov-report=term-missing

check: lint typecheck test  ## What CI runs

run:       ## The app alone, in-memory storage (needs a gateway at MONEY_GATEWAY_URL)
	MONEY_LOG_JSON=false money-agent

up:        ## Whole stack: Postgres + Groundwire + app on http://localhost:8100
	docker compose up --build

down:
	docker compose down
