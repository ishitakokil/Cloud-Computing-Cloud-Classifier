.PHONY: build run test lint

# Build the Docker image
build:
	docker build -f dockerfiles/Dockerfile -t cloud-pipeline .

# Run the full pipeline with AWS credentials if needed
run:
	docker run \
		-v $(shell pwd):/app \
		-v ~/.aws:/root/.aws \
		cloud-pipeline

# Run unit tests inside Docker
test:
	docker run \
		-v $(shell pwd):/app \
		-v ~/.aws:/root/.aws \
		cloud-pipeline pytest tests/

# Run pylint for style checks 
lint:
	pylint src tests pipeline.py
