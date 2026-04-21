#!/usr/bin/env python3
"""
rollback.py — point the 'production' alias back to the previous version.

Usage:
    python rollback.py                    # rolls back to previous production version
    python rollback.py --version 3        # rolls back to a specific version
"""

import os
import json
import argparse
import urllib.request
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.MlflowClient()

# Must match train.py / promote.py
MODEL = "bart-meeting-summarizer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()

    # Find current production version
    try:
        current = client.get_model_version_by_alias(MODEL, "production")
        current_v = int(current.version)
        print(f"Current production: v{current_v}")
    except Exception:
        current_v = None
        print("No production alias set yet")

    # Determine rollback target
    if args.version is not None:
        target_v = args.version
    elif current_v and current_v > 1:
        target_v = current_v - 1
    else:
        print("Nothing to roll back to")
        return 0

    # Point production alias to target
    client.set_registered_model_alias(MODEL, "production", target_v)
    print(f"Rolled back: production -> v{target_v}")

    # Notify inference server to reload
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
            print(f"Warning: rollback succeeded, but reload failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
