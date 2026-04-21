FROM python:3.12-slim

LABEL maintainer="jb9202@nyu.edu"
LABEL description="Data pipeline for MeetingBank transcription service - proj30"

# System dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all data scripts
COPY scripts/            ./scripts/
COPY quality/            ./quality/
COPY feature_extraction/ ./feature_extraction/

# Add feature_extraction to Python path so imports work
ENV PYTHONPATH=/app/feature_extraction:/app/scripts

# Chameleon credentials — pass at runtime, never bake in
ENV OS_AUTH_URL=""
ENV OS_APPLICATION_CREDENTIAL_ID=""
ENV OS_APPLICATION_CREDENTIAL_SECRET=""
ENV OS_REGION_NAME="CHI@TACC"

# Points to shared team MLflow server
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"

ENV CONTAINER_NAME="ObjStore_projecttranscriptionmirotalk"
ENV DATASET_VERSION="1.0"

CMD ["python3", "scripts/run_pipeline.py", "--help"]