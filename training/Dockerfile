# Base image already has CUDA drivers + PyTorch — matches your H100 node
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY train.py .

# /mnt/data is mounted at runtime from the host for adapter output + checkpoints
# baseline_scores.json is passed in at runtime via volume mount too
ENTRYPOINT ["python", "train.py"]