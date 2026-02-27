# ═══════════════════════════════════════════════
# CropFresh AI — Makefile
# Quick commands for development, testing, deployment
# ═══════════════════════════════════════════════

.PHONY: dev test lint deploy setup clean eval scrape

# ─── Development ───────────────────────────────

dev:  ## Start development server
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

setup:  ## Set up development environment
	uv sync --all-extras
	cp -n .env.example .env 2>/dev/null || true
	@echo "✅ Dev environment ready! Edit .env with your API keys."

# ─── Testing ───────────────────────────────────

test:  ## Run all tests
	uv run pytest tests/ -v --tb=short

test-unit:  ## Run unit tests only
	uv run pytest tests/ -v --tb=short -m "not integration and not e2e"

test-integration:  ## Run integration tests
	uv run pytest tests/integration/ -v --tb=short

test-e2e:  ## Run end-to-end tests
	uv run pytest tests/e2e/ -v --tb=short

test-load:  ## Run load tests with Locust
	uv run locust -f tests/load/locustfile.py

# ─── Code Quality ─────────────────────────────

lint:  ## Lint code with ruff
	uv run ruff check src/ ai/ tests/ scripts/

lint-fix:  ## Auto-fix lint issues
	uv run ruff check src/ ai/ tests/ scripts/ --fix

format:  ## Format code with ruff
	uv run ruff format src/ ai/ tests/ scripts/

typecheck:  ## Run type checking
	uv run mypy src/ --ignore-missing-imports

# ─── AI & Agents ──────────────────────────────

eval:  ## Run agent evaluations
	uv run python ai/evals/run_evals.py

eval-crop:  ## Evaluate crop listing agent
	uv run python ai/evals/run_evals.py --agent crop_listing

eval-price:  ## Evaluate price prediction agent
	uv run python ai/evals/run_evals.py --agent price_prediction

# ─── Data Pipeline ────────────────────────────

scrape:  ## Run all scrapers
	uv run python scripts/run-all-evals.sh

seed:  ## Seed development database
	uv run python scripts/seed-database.py

# ─── Docker ───────────────────────────────────

docker-up:  ## Start all services with Docker
	docker-compose up -d

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## View service logs
	docker-compose logs -f

# ─── Deployment ───────────────────────────────

deploy-staging:  ## Deploy to staging (GCP Cloud Run)
	@echo "Deploying to staging..."
	gcloud run deploy cropfresh-staging --source .

deploy-prod:  ## Deploy to production
	@echo "⚠️  Deploying to PRODUCTION..."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	gcloud run deploy cropfresh-prod --source .

# ─── Utilities ────────────────────────────────

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache build dist *.egg-info

daily-log:  ## Generate today's daily dev log
	uv run python scripts/generate-daily-log.py

cost-report:  ## Generate cost/token usage report
	uv run python scripts/cost-report.py

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
