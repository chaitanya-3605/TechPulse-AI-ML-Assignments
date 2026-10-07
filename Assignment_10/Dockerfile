# ==============================================================================
# Tata Technologies TechPulse Applied AI/ML Laboratory - Assignment 10
# Student: Chaitanya Pahilwan
# Lightweight Container Image for Automotive MLOps Inference Service
# ==============================================================================

FROM python:3.11-slim

LABEL maintainer="Chaitanya Pahilwan"
LABEL project="Tata Technologies TechPulse Applied AI/ML"
LABEL assignment="Assignment 10 - MLOps Workflow Simulation"

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bundle serialized model artifact and inference driver
COPY rf_model.joblib .
COPY predict.py .

# Define container runtime entrypoint
CMD ["python", "predict.py"]
