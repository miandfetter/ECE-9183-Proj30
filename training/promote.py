#!/usr/bin/env python3
"""
promote.py — promotes the current staging model to production.

Usage:
    python3 promote.py                    # promotes current staging to production
    python3 promote.py --version 3        # promotes a specific version to production

Requirements:
    MLFLOW_TRACKING_URI env var must be set
"""
import os
import json
import argparse
import urllib.request
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.MlflowClient()
MODEL  = "bart-meeting-summarizer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=None,
                        help="Specific version to promote (default: current staging)")
    args = parser.parse_args()

    # Find version to promote
    if args.version:
        target_v = args.version
    else:
        try:
            mv = client.get_model_version_by_alias(MODEL, "staging")
            target_v = int(mv.version)
            print(f"Current staging: v{target_v} (rougeL: {_get_rougeL(mv.run_id):.4f})")
        except Exception as e:
            print(f"No staging model found: {e}")
            return

    # Check if there's already a production model
    try:
        current_prod = client.get_model_version_by_alias(MODEL, "production")
        print(f"Current production: v{current_prod.version} (rougeL: {_get_rougeL(current_prod.run_id):.4f})")
    except Exception:
        print("No current production model")

    # Confirm
    print(f"\nPromoting v{target_v} to production...")
    client.set_registered_model_alias(MODEL, "production", target_v)
    print(f"Done — production -> v{target_v}")

    # Notify inference server
    deploy_url = os.environ.get("INFERENCE_SERVER_URL")
    if deploy_url:
        try:
            req = urllib.request.Request(
                f"{deploy_url}/reload",
                data=json.dumps({"alias": "production", "version": target_v}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"Inference server reloaded: {resp.status}")
        except Exception as e:
            print(f"Inference server reload failed (non-fatal): {e}")
    else:
        print("INFERENCE_SERVER_URL not set — skipping inference server reload")


def _get_rougeL(run_id):
    try:
        run = client.get_run(run_id)
        return float(run.data.metrics.get("rougeL", 0))
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()