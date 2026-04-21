#!/usr/bin/env python3
"""
rollback.py — point the 'production' alias back to the previous version.

Usage:
    python rollback.py                    # rolls back to previous production version
    python rollback.py --version 3        # rolls back to a specific version
"""
import os
import argparse
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.MlflowClient()
MODEL  = "bart-mirotalk-summarizer"

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
    if args.version:
        target_v = args.version
    elif current_v and current_v > 1:
        target_v = current_v - 1
    else:
        print("Nothing to roll back to")
        return

    client.set_registered_model_alias(MODEL, "production", target_v)
    print(f"Rolled back: production -> v{target_v}")

    # Tell inference server to reload
    deploy_url = os.environ.get("INFERENCE_SERVER_URL")
    if deploy_url:
        import urllib.request, json
        req = urllib.request.Request(
            f"{deploy_url}/reload",
            data=json.dumps({"alias": "production", "version": target_v}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"Inference server reloaded: {resp.status}")

if __name__ == "__main__":
    main()