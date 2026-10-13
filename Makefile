.PHONY: install lint typecheck test cov check run docker-build up down chaos

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

chaos:            ## Fire 500 requests at the flaky demo route
	python scripts/chaos_demo.py --requests 500 --concurrency 50
