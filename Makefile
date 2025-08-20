.PHONY: help install test run docker-build docker-run clean lint format security

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
IMAGE_NAME := heart-attack-api
VERSION := $(shell git describe --tags --always --dirty)

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make test          Run tests with coverage"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black"
	@echo "  make security      Run security checks"
	@echo "  make run           Run application locally"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run with Docker Compose"
	@echo "  make docker-stop   Stop Docker Compose services"
	@echo "  make clean         Clean up generated files"
	@echo "  make migrate       Run database migrations"
	@echo "  make train         Train a new model"

# Development setup
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	mkdir -p logs models data/raw data/processed data/uploads data/predictions

requirements-dev.txt:
	@echo "pytest==7.4.3" > requirements-dev.txt
	@echo "pytest-cov==4.1.0" >> requirements-dev.txt
	@echo "pytest-asyncio==0.21.1" >> requirements-dev.txt
	@echo "black==23.11.0" >> requirements-dev.txt
	@echo "flake8==6.1.0" >> requirements-dev.txt
	@echo "mypy==1.7.0" >> requirements-dev.txt
	@echo "bandit==1.7.5" >> requirements-dev.txt
	@echo "pre-commit==3.5.0" >> requirements-dev.txt

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-integration:
	pytest tests/integration/ -v -m integration

test-watch:
	pytest-watch tests/ -v

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	mypy src/ --ignore-missing-imports
	bandit -r src/ -ll

format:
	black src/ tests/
	isort src/ tests/

security:
	bandit -r src/ -f json -o security-report.json
	safety check --json > safety-report.json || true
	@echo "Security reports generated: security-report.json, safety-report.json"

# Local development
run:
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

run-prod:
	$(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# Docker operations
docker-build:
	$(DOCKER) build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest .

docker-run: docker-build
	$(DOCKER_COMPOSE) up -d

docker-stop:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f api

docker-clean:
	$(DOCKER_COMPOSE) down -v
	$(DOCKER) system prune -f

# Database operations
migrate:
	alembic upgrade head

migrate-create:
	@read -p "Enter migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

migrate-rollback:
	alembic downgrade -1

# Model operations
train:
	$(PYTHON) scripts/train_model.py

train-auto:
	curl -X POST http://localhost:8000/api/v1/train/ \
		-H "Content-Type: application/json" \
		-d '{"model_type": "random_forest", "auto_tune": true}'

# Deployment
deploy-k8s:
	kubectl apply -f kubernetes/deployment.yaml
	kubectl rollout status deployment/heart-attack-api -n heart-attack-prediction

deploy-helm:
	helm upgrade --install heart-attack-api ./helm/heart-attack-api \
		--namespace heart-attack-prediction --create-namespace

# Monitoring
logs:
	tail -f logs/app.log

metrics:
	curl http://localhost:8000/metrics

health:
	curl http://localhost:8000/api/v1/health/detailed | jq

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -rf build/ dist/ *.egg-info
	rm -f security-report.json safety-report.json

# Utilities
version:
	@echo $(VERSION)

check-env:
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"
	@echo "Docker version: $(shell $(DOCKER) --version)"
	@echo "Docker Compose version: $(shell $(DOCKER_COMPOSE) --version)"

.env:
	cp .env.example .env
	@echo "Created .env file from .env.example"
	@echo "Please update the values in .env file"