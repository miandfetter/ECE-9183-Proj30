#!/usr/bin/env python3


import argparse
import json
import os
from datetime import datetime, timezone, timedelta

import mlflow
import swiftclient
from keystoneauth1 import session
from keystoneauth1.identity.v3 import ApplicationCredential


# ── Quality thresholds ────────────────────────────────────────────────────────
MIN_TRANSCRIPT_LENGTH = 10    # ignore very short transcripts
MAX_TRANSCRIPT_LENGTH = 10000 # ignore abnormally long ones
MIN_LATENCY_MS        = 10    # ignore suspiciously fast responses


def get_conn():
    """Connect using keystoneauth application credentials."""
    auth = ApplicationCredential(
        auth_url=os.environ["OS_AUTH_URL"],
        application_credential_id=os.environ.get("OS_APPLICATION_CREDENTIAL_ID"),
        application_credential_secret=os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET"),
    )
    sess = session.Session(auth=auth)
    token = sess.get_token()
    storage_url = sess.get_endpoint(
        service_type="object-store",
        interface="public",
        region_name=os.environ.get("OS_REGION_NAME", "CHI@TACC")
    )
    return swiftclient.Connection(
        preauthtoken=token,
        preauthurl=storage_url,
    )


def passes_quality_gate(log, min_confidence):
    """Check if an inference log meets quality standards for retraining."""
    transcript  = log.get("transcript", "")
    confidence  = log.get("confidence", 0.0)
    latency_ms  = log.get("latency_ms", 0.0)

    if not transcript or len(transcript) < MIN_TRANSCRIPT_LENGTH:
        return False, "transcript_too_short"

    if len(transcript) > MAX_TRANSCRIPT_LENGTH:
        return False, "transcript_too_long"

    if confidence < min_confidence:
        return False, f"low_confidence:{confidence:.2f}"

    if latency_ms < MIN_LATENCY_MS:
        return False, "suspiciously_fast"

    return True, "ok"


def collect_feedback(container, min_confidence, days, mlflow_uri):
    conn = get_conn()

    print(f"\n{'='*55}")
    print(f"  FEEDBACK COLLECTION PIPELINE")
    print(f"{'='*55}")
    print(f"  Container      : {container}")
    print(f"  Min confidence : {min_confidence}")
    print(f"  Window         : last {days} days")
    print(f"{'='*55}\n")

    collected = []
    rejected  = []
    total_logs = 0

    # Scan inference logs from last N days
    for day_offset in range(days):
        date = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        prefix = f"inference_logs/{date}/"

        try:
            _, objects = conn.get_container(
                container, prefix=prefix, full_listing=True)

            if not objects:
                print(f"  {date}: no logs found")
                continue

            print(f"  {date}: {len(objects)} logs found")

            for obj in objects:
                try:
                    _, body = conn.get_object(container, obj["name"])
                    log = json.loads(body)
                    total_logs += 1

                    passed, reason = passes_quality_gate(log, min_confidence)

                    if passed:
                        collected.append({
                            "meeting_id"    : f"inference_{log.get('request_id', 'unknown')}",
                            "transcript"    : log.get("transcript", ""),
                            "confidence"    : log.get("confidence", 0.0),
                            "duration_sec"  : log.get("duration_sec", 0.0),
                            "source"        : "inference",
                            "model_version" : log.get("model_version", "unknown"),
                            "timestamp"     : log.get("timestamp"),
                            "is_synthetic"  : False,
                            "split"         : "train",
                        })
                    else:
                        rejected.append({
                            "request_id": log.get("request_id"),
                            "reason"    : reason,
                        })

                except Exception as e:
                    print(f"    Warning: could not read {obj['name']}: {e}")
                    continue

        except Exception as e:
            print(f"  {date}: could not list logs — {e}")
            continue

    print(f"\n  Total logs scanned : {total_logs}")
    print(f"  Collected          : {len(collected)}")
    print(f"  Rejected           : {len(rejected)}")

    if not collected:
        print("\n  No feedback collected yet.")
        print("  This is expected if serving is not yet running or no traffic.")
        print("  Drift monitor will trigger retraining when traffic starts.")

        # Still log to MLflow so there's a record
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("data_quality_feedback")
        with mlflow.start_run(run_name=f"feedback_collection_empty"):
            mlflow.log_metric("total_logs",       0)
            mlflow.log_metric("collected",        0)
            mlflow.log_metric("rejected",         0)
            mlflow.log_metric("collection_rate",  0.0)
            mlflow.set_tag("status", "no_data")
        return

    # Upload collected feedback to object storage
    timestamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    feedback_path = f"feedback/collected_{timestamp}.json"
    collection_rate = len(collected) / max(total_logs, 1)

    feedback_manifest = {
        "collected_at"    : datetime.now(timezone.utc).isoformat(),
        "window_days"     : days,
        "min_confidence"  : min_confidence,
        "total_logs"      : total_logs,
        "record_count"    : len(collected),
        "rejected_count"  : len(rejected),
        "collection_rate" : round(collection_rate, 4),
        "records"         : collected,
        "rejection_sample": rejected[:10],
    }

    conn.put_object(
        container,
        feedback_path,
        json.dumps(feedback_manifest, indent=2).encode(),
        content_type="application/json"
    )

    print(f"\n  Saved to   : {feedback_path}")
    print(f"  Collection rate : {collection_rate:.1%}")
    print(f"\n  These records will be included in next retraining cycle.")
    print(f"  Pass this to batch_pipeline.py as additional training data.")

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("data_quality_feedback")

    with mlflow.start_run(run_name=f"feedback_collection_{timestamp}"):
        mlflow.log_param("container",       container)
        mlflow.log_param("min_confidence",  min_confidence)
        mlflow.log_param("window_days",     days)

        mlflow.log_metric("total_logs",     total_logs)
        mlflow.log_metric("collected",      len(collected))
        mlflow.log_metric("rejected",       len(rejected))
        mlflow.log_metric("collection_rate",collection_rate)

        mlflow.set_tag("feedback_path",     feedback_path)
        mlflow.set_tag("status",            "collected")

    print(f"\n  MLflow run logged to experiment: data_quality_feedback")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect high-quality inference transcripts for retraining"
    )
    parser.add_argument("--container",       required=True)
    parser.add_argument("--min-confidence",  type=float, default=0.85,
                        help="Minimum model confidence to include (default: 0.85)")
    parser.add_argument("--days",            type=int,   default=7,
                        help="Number of past days to collect from (default: 7)")
    parser.add_argument("--mlflow-uri",      default=os.environ.get(
                         "MLFLOW_TRACKING_URI", "file:///work/mlruns"))
    args = parser.parse_args()

    collect_feedback(
        container       = args.container,
        min_confidence  = args.min_confidence,
        days            = args.days,
        mlflow_uri      = args.mlflow_uri,
    )
