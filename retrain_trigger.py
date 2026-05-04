#!/usr/bin/env python3
"""
retrain_trigger.py — runs on the MLflow VM via cron.
SSHes into an already-running GPU node and fires training container.

Cron entry:
    0 2 * * 0 set -a; source /home/cc/.mlflow_s3_credentials; set +a;
    python3 /home/cc/proj30/training/retrain_trigger.py >> /home/cc/retrain.log 2>&1
"""
import os
import sys
import subprocess
from datetime import datetime

GPU_NODE_IP     = os.environ["GPU_NODE_IP"]
SSH_KEY         = os.path.expanduser("~/.ssh/id_rsa_chameleon")
REMOTE_USER     = "cc"

MLFLOW_URI      = os.environ["MLFLOW_TRACKING_URI"]
S3_ENDPOINT     = os.environ["MLFLOW_S3_ENDPOINT_URL"]
AWS_KEY         = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET      = os.environ["AWS_SECRET_ACCESS_KEY"]
TRAINING_BUCKET = os.environ["TRAINING_DATA_BUCKET"]
INFERENCE_URL   = os.environ.get("INFERENCE_SERVER_URL", "")


def log(msg):
    print(f"[{datetime.now().isoformat()}] {msg}", flush=True)



def main():
    log("=== Retrain trigger started ===")

    docker_cmd = (
        f"docker run --rm --gpus all "
        f"-e MLFLOW_TRACKING_URI={MLFLOW_URI} "
        f"-e MLFLOW_S3_ENDPOINT_URL={S3_ENDPOINT} "
        f"-e AWS_ACCESS_KEY_ID={AWS_KEY} "
        f"-e AWS_SECRET_ACCESS_KEY={AWS_SECRET} "
        f"-e BUCKET_NAME={TRAINING_BUCKET} "
        f"-e DATA_PREFIX=raw/v1.0 "
        f"-e INFERENCE_SERVER_URL={INFERENCE_URL} "
        f"-v /mnt/data:/mnt/data "
        f"proj30-train:latest"
    )

    ssh_cmd = [
        "ssh", "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{REMOTE_USER}@{GPU_NODE_IP}",
        docker_cmd,
    ]

    log(f"SSHing into GPU node {GPU_NODE_IP}...")
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        log(f"Training failed:\n{result.stderr}")
        sys.exit(1)

    log("Training succeeded")
    log("=== Retrain trigger complete ===")


if __name__ == "__main__":
    main()