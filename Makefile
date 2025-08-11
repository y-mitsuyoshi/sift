# Makefile for liveness-api

# Default command
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make up      - Start the services in the background"
	@echo "  make down    - Stop and remove the services"
	@echo "  make logs    - Follow the logs of the API server"
	@echo "  make test    - Run the API test script"
	@echo "  make shell   - Get a shell inside the API container"
	@echo "  make build   - Force a rebuild of the Docker images"

# Build the Docker images
.PHONY: build
build:
	docker compose build

# Start services in the background
.PHONY: up
up:
	docker compose up -d --build

# Stop and remove services
.PHONY: down
down:
	docker compose down

# Follow logs
.PHONY: logs
logs:
	docker compose logs -f liveness-api

# Run the API test script
# Note: This requires 'test_video.mp4' to be present in the root directory.
.PHONY: test
test:
	python3 test_api.py test_video.mp4

# Get a shell inside the running container
.PHONY: shell
shell:
	docker compose exec liveness-api /bin/bash
