.PHONY: help install dev-install lint test generate-data docker-up docker-down bench clean

SHELL := /bin/bash
PROJECT_ROOT := $(shell pwd)
PYTHON := python3
DATA_DIR := $(PROJECT_ROOT)/data

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────────────

install: ## Install production dependencies
	$(PYTHON) -m pip install -e .

dev-install: ## Install with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

gpu-install: ## Install with GPU support
	$(PYTHON) -m pip install -e ".[dev,gpu]"

full-install: ## Install everything
	$(PYTHON) -m pip install -e ".[dev,gpu,rl,llm]"

# ─── Code Quality ────────────────────────────────────────────────────────────

lint: ## Run linters
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy services/ ml/ --ignore-missing-imports

format: ## Auto-format code
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test: ## Run all tests
	$(PYTHON) -m pytest -v --tb=short

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest -v --cov=services --cov=ml --cov-report=html

# ─── Data Generation ─────────────────────────────────────────────────────────

generate-data-small: ## Generate small dataset (100K interactions)
	$(PYTHON) -m ml.data_simulator.cli \
		--num-users 5000 --num-videos 10000 --num-interactions 100000 \
		--output-dir $(DATA_DIR)/small

generate-data-medium: ## Generate medium dataset (1M interactions)
	$(PYTHON) -m ml.data_simulator.cli \
		--num-users 20000 --num-videos 50000 --num-interactions 1000000 \
		--output-dir $(DATA_DIR)/medium

generate-data-large: ## Generate large dataset (10M interactions)
	$(PYTHON) -m ml.data_simulator.cli \
		--num-users 50000 --num-videos 100000 --num-interactions 10000000 \
		--output-dir $(DATA_DIR)/large

# ─── Infrastructure ──────────────────────────────────────────────────────────

docker-up: ## Start local infrastructure
	docker compose up -d

docker-down: ## Stop local infrastructure
	docker compose down

docker-build: ## Build all service images
	docker compose build

docker-logs: ## Tail infrastructure logs
	docker compose logs -f

kafka-topics: ## Create Kafka topics
	docker compose exec kafka kafka-topics --create --bootstrap-server localhost:9092 \
		--replication-factor 1 --partitions 12 --topic user.events.raw || true
	docker compose exec kafka kafka-topics --create --bootstrap-server localhost:9092 \
		--replication-factor 1 --partitions 12 --topic user.events.enriched || true
	docker compose exec kafka kafka-topics --create --bootstrap-server localhost:9092 \
		--replication-factor 1 --partitions 6 --topic content.metadata || true
	docker compose exec kafka kafka-topics --create --bootstrap-server localhost:9092 \
		--replication-factor 1 --partitions 12 --topic impressions.log || true
	docker compose exec kafka kafka-topics --create --bootstrap-server localhost:9092 \
		--replication-factor 1 --partitions 6 --topic recommendations.served || true
	docker compose exec kafka kafka-topics --create --bootstrap-server localhost:9092 \
		--replication-factor 1 --partitions 3 --topic model.signals || true

# ─── Services ────────────────────────────────────────────────────────────────

run-gateway: ## Run API gateway locally
	$(PYTHON) -m uvicorn services.api_gateway.main:app --host 0.0.0.0 --port 8000 --reload

run-retrieval: ## Run retrieval service locally
	$(PYTHON) -m uvicorn services.retrieval.main:app --host 0.0.0.0 --port 8003 --reload

run-ranking: ## Run ranking service locally
	$(PYTHON) -m uvicorn services.ranking.main:app --host 0.0.0.0 --port 8004 --reload

# ─── Benchmarks ──────────────────────────────────────────────────────────────

bench-retrieval: ## Benchmark retrieval latency
	$(PYTHON) -m benchmarks.latency_bench --service retrieval --num-requests 1000

bench-ranking: ## Benchmark ranking latency
	$(PYTHON) -m benchmarks.latency_bench --service ranking --num-requests 1000

bench-e2e: ## End-to-end benchmark
	$(PYTHON) -m benchmarks.throughput_bench --duration 60

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache htmlcov .mypy_cache

clean-data: ## Remove generated datasets
	rm -rf $(DATA_DIR)
