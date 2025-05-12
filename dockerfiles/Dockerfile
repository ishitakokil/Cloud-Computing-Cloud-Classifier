# Use official Python slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy everything from your project root to /app
COPY . .
COPY config/ config/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# By default, run the pipeline
CMD ["python", "pipeline.py", "--config", "config/default-config.yaml"]