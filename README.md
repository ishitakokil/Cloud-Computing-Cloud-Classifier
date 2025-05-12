# MLDS 423 Cloud Engineering Homework 2

This project implements a cloud-native, end-to-end machine learning pipeline to classify cloud types using satellite-based atmospheric data. The system handles data acquisition, feature engineering, exploratory data analysis (EDA), model training, evaluation, and artifact upload to AWS S3 — all inside a Dockerized, reproducible environment.

The pipeline is modular, reproducible, and testable, following software engineering best practices. It is configured using a YAML file and runs with a single `make run` command.

## Project Structure

- **`pipeline.py`**  
  Main script orchestrating the entire ML pipeline from data acquisition to evaluation and optional S3 upload.

- **`config/default-config.yaml`**  
  Centralized configuration for paths, model parameters, feature engineering, AWS, and more.

- **`src/`** – Modularized scripts for each pipeline stage:
  - `acquire_data.py` – Downloads data from a URL.
  - `create_dataset.py` – Loads and cleans the raw dataset.
  - `generate_features.py` – Engineers additional features and labels.
  - `train_model.py` – Trains a RandomForest classifier and splits data.
  - `score_model.py` – Predicts test results and formats outputs.
  - `evaluate_performance.py` – Calculates evaluation metrics and plots ROC.
  - `analysis.py` – Saves class-wise histograms (EDA).
  - `aws_utils.py` – Uploads pipeline artifacts to AWS S3.

- **`tests/`**  
  Contains unit tests for feature generation and error handling (happy & unhappy paths).

- **`dockerfiles/Dockerfile`**  
  Defines the environment to run the pipeline reproducibly in a container.

- **`Makefile`**  
  Provides commands to:
  - `make build` – Build the Docker image  
  - `make run` – Run the full pipeline  
  - `make test` – Run unit tests inside Docker  
  - `make lint` – Check code with `pylint`

- **`models/` & `runs/`**  
  Store trained model, metrics, ROC curves, and timestamped run artifacts.

## Setup
### Cloning the git repo
- `git clone https://github.com/NUMLDS/423-2025-hw2-jji9639`

### Docker Installation
To run this project in a containerized environment, make sure Docker is installed:
#### **Mac/Linux:**

1. [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Verify installation:
   ```bash
   docker --version

`sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker`

## How to Run the Pipeline
### Build the Docker Image
`make build` 
### Run the Full Pipeline
`make run`
  
This will:
- Download and process the cloud dataset
- Generate features and labels
- Train a model and evaluate it
- Save artifacts to runs/cloud-classifier-pipeline_<timestamp>/

## Running Unit Tests
`make test`

This runs all tests in the tests/ directory using pytest.

## Code Style Checks
`make lint`

Use pylint to check that all code follows PEP8 and passes linting. 

## AWS S3 Upload

### Set up AWS

To enable AWS uploads during `make run`, follow these steps:

####  1. Install AWS CLI

[Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) and verify:

```
aws --version
```

#### 2. Configure AWS Credentials
`aws configure`

You’ll be prompted to enter:

- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., us-east-2)
- Output format (optional, e.g., json)

Credentials are stored in ~/.aws/credentials and are used by the makefile which mounts your local AWS config into the container.

### Enable Upload in the Config
- In config/default-config.yaml: 

`aws:
  upload: true
  bucket_name: your-bucket-name
  prefix: cloud_engineering
`
- Authenticate locally using:
`aws configure`

- Run the pipeline:
`make run`

- Artifacts will upload to:
`s3://your-s3-bucket-name/cloud_engineering/<all files in runs/...>`