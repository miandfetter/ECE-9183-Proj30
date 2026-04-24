#!/usr/bin/env python3
import argparse, json, os, subprocess
from datetime import datetime, timezone, timedelta
import mlflow

MIN_TRANSCRIPT_LENGTH = 10
MAX_TRANSCRIPT_LENGTH = 10000
MIN_LATENCY_MS = 10

def get_conn():
    import swiftclient
    return swiftclient.Connection(
        auth_version="3",
        authurl=os.environ["OS_AUTH_URL"],
        os_options={
            "auth_type": "v3applicationcredential",
            "application_credential_id": os.environ.get("OS_APPLICATION_CREDENTIAL_ID"),
            "application_credential_secret": os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET"),
            "region_name": os.environ.get("OS_REGION_NAME", "CHI@TACC"),
        },
    )

def passes_quality_gate(log, min_confidence):
    transcript = log.get("transcript", "")
    confidence = log.get("confidence", 0.0)
    latency_ms = log.get("latency_ms", 0.0)
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
    rejected = []
    total_logs = 0

    for day_offset in range(days):
        date = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        prefix = f"inference_logs/{date}/"
        try:
            _, objects = conn.get_container(container, prefix=prefix, full_listing=True)
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
                            "meeting_id"   : f"inference_{log.get('request_id', 'unknown')}",
                            "transcript"   : log.get("transcript", ""),
                            "confidence"   : log.get("confidence", 0.0),
                            "duration_sec" : log.get("duration_sec", 0.0),
                            "source"       : "inference",
                            "model_version": log.get("model_version", "unknown"),
                            "timestamp"    : log.get("timestamp"),
                            "is_synthetic" : False,
                            "split"        : "train",
                        })
                    else:
                        rejected.append({"request_id": log.get("request_id"), "reason": reason})
                except Exception:
                    continue
        except Exception as e:
            print(f"  {date}: could not list logs — {e}")
            continue

    print(f"\n  Total logs scanned : {total_logs}")
    print(f"  Collected          : {len(collected)}")
    print(f"  Rejected           : {len(rejected)}")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("data_quality_feedback")

    if not collected:
        print("\n  No feedback collected yet — serving not running or no traffic")
        with mlflow.start_run(run_name="feedback_collection_empty"):
            mlflow.log_metric("total_logs", 0)
            mlflow.log_metric("collected", 0)
            mlflow.set_tag("status", "no_data")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    feedback_path = f"feedback/collected_{timestamp}.json"
    collection_rate = len(collected) / max(total_logs, 1)

    conn.put_object(
        container, feedback_path,
        json.dumps({
            "collected_at"   : datetime.now(timezone.utc).isoformat(),
            "window_days"    : days,
            "min_confidence" : min_confidence,
            "total_logs"     : total_logs,
            "record_count"   : len(collected),
            "rejected_count" : len(rejected),
            "collection_rate": round(collection_rate, 4),
            "records"        : collected,
        }, indent=2).encode(),
        content_type="application/json"
    )

    print(f"\n  Saved to        : {feedback_path}")
    print(f"  Collection rate : {collection_rate:.1%}")

    with mlflow.start_run(run_name=f"feedback_collection_{timestamp}"):
        mlflow.log_param("container",        container)
        mlflow.log_param("min_confidence",   min_confidence)
        mlflow.log_param("window_days",      days)
        mlflow.log_metric("total_logs",      total_logs)
        mlflow.log_metric("collected",       len(collected))
        mlflow.log_metric("rejected",        len(rejected))
        mlflow.log_metric("collection_rate", collection_rate)
        mlflow.set_tag("feedback_path",      feedback_path)
        mlflow.set_tag("status",             "collected")

    print(f"  MLflow logged to: data_quality_feedback")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container",      required=True)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--days",           type=int,   default=7)
    parser.add_argument("--mlflow-uri",     default=os.environ.get(
                         "MLFLOW_TRACKING_URI", "http://129.114.27.29:8002"))
    args = parser.parse_args()
    collect_feedback(args.container, args.min_confidence, args.days, args.mlflow_uri)
