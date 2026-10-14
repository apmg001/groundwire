.PHONY: install lint typecheck test cov check run docker-build up down chaos demo

install:          ## Install the package with dev tools
	pip install -e ".[dev]"

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
