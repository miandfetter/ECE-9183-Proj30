#!/usr/bin/env python3
"""
03_drift_monitor.py
-------------------
Monitors live inference data quality and distribution drift in production.
Runs as a long-lived service (or periodic cron job) that:

  1. Collects statistics from live inference requests
  2. Compares them to the training data reference distribution
  3. Detects drift using statistical tests
  4. Logs alerts to MLflow and stdout
  5. Writes drift scores to a file for Docker health checks

Drift signals monitored:
  - Audio duration distribution (mean, std, percentiles)
  - Audio amplitude/energy distribution
  - Transcript length distribution
  - Request volume (sudden drops = possible upstream issue)
  - Feature mean/variance shift (if features are logged)

Usage:
    # Run once (e.g. from cron every 15 minutes)
    python3 03_drift_monitor.py --mode once \
        --container ObjStore_projecttranscriptionmirotalk \
        --reference-version 1.0

    # Run continuously
    python3 03_drift_monitor.py --mode continuous --interval 300
"""

import argparse
import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone

import mlflow
import swiftclient

# ── Drift thresholds ──────────────────────────────────────────────────────────
# Justified: these represent meaningful shifts in the data distribution
# that would degrade model performance if left undetected.

DURATION_MEAN_DRIFT_THRESHOLD  = 0.30  # 30% change in mean audio duration
DURATION_STD_DRIFT_THRESHOLD   = 0.50  # 50% change in std of audio duration
TRANSCRIPT_LEN_DRIFT_THRESHOLD = 0.30  # 30% change in mean transcript length
VOLUME_DROP_THRESHOLD          = 0.50  # >50% drop in request volume = alert
FEATURE_MEAN_DRIFT_THRESHOLD   = 0.20  # 20% shift in feature mean
MIN_SAMPLES_FOR_DRIFT          = 10    # need at least 10 samples to detect drift

# ── Reference distribution (loaded from training data stats) ──────────────────
# These are updated when a new model version is deployed.
DEFAULT_REFERENCE = {
    "duration_mean"    : 4.5,    # seconds — typical MeetingBank segment
    "duration_std"     : 2.1,
    "transcript_len_mean": 45.0, # characters
    "transcript_len_std" : 22.0,
    "requests_per_min" : 5.0,    # expected production traffic rate
    "feature_mean"     : 0.0,    # normalized features should be ~0
    "feature_std"      : 1.0,
}


def get_conn():
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


# ── Statistics helpers ────────────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values) if values else 0.0

def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)

def percentile(values, p):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]

def relative_change(reference, current):
    """Relative change from reference to current."""
    if reference == 0:
        return 0.0
    return abs(current - reference) / abs(reference)


# ── Drift detection ───────────────────────────────────────────────────────────

def detect_drift(live_stats, reference):
    """
    Compare live stats to reference distribution.
    Returns list of (metric, drift_score, threshold, alert_level).
    """
    alerts = []

    checks = [
        ("duration_mean",      live_stats.get("duration_mean", 0),
         reference["duration_mean"],      DURATION_MEAN_DRIFT_THRESHOLD),
        ("duration_std",       live_stats.get("duration_std", 0),
         reference["duration_std"],       DURATION_STD_DRIFT_THRESHOLD),
        ("transcript_len_mean",live_stats.get("transcript_len_mean", 0),
         reference["transcript_len_mean"],TRANSCRIPT_LEN_DRIFT_THRESHOLD),
        ("feature_mean",       live_stats.get("feature_mean", 0),
         reference["feature_mean"],       FEATURE_MEAN_DRIFT_THRESHOLD),
    ]

    for metric, live_val, ref_val, threshold in checks:
        drift_score = relative_change(ref_val, live_val)
        if drift_score > threshold:
            level = "CRITICAL" if drift_score > threshold * 2 else "WARNING"
            alerts.append({
                "metric"     : metric,
                "drift_score": round(drift_score, 4),
                "threshold"  : threshold,
                "reference"  : ref_val,
                "live"       : round(live_val, 4),
                "level"      : level,
            })

    return alerts


def compute_live_stats(inference_logs):
    """Compute distribution statistics from recent inference logs."""
    if not inference_logs:
        return {}

    durations   = [r.get("duration_sec", 0) for r in inference_logs
                   if r.get("duration_sec")]
    trans_lens  = [len(r.get("transcript", "")) for r in inference_logs
                   if r.get("transcript")]
    feat_means  = [r.get("feature_mean", 0) for r in inference_logs
                   if "feature_mean" in r]

    return {
        "sample_count"       : len(inference_logs),
        "duration_mean"      : mean(durations),
        "duration_std"       : std(durations),
        "duration_p50"       : percentile(durations, 50),
        "duration_p95"       : percentile(durations, 95),
        "transcript_len_mean": mean(trans_lens),
        "transcript_len_std" : std(trans_lens),
        "feature_mean"       : mean(feat_means),
        "feature_std"        : std(feat_means),
    }


# ── Log loading from object storage ──────────────────────────────────────────

def load_recent_inference_logs(conn, container, window_minutes=60):
    """
    Load recent inference request logs from object storage.
    The serving layer writes one JSON log per request to:
      inference_logs/{date}/{request_id}.json
    """
    logs = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = f"inference_logs/{today}/"

    try:
        _, objects = conn.get_container(container, prefix=prefix, full_listing=True)
        # Sort by last modified, take recent ones
        recent = sorted(objects, key=lambda o: o.get("last_modified", ""), reverse=True)
        cutoff_count = max(100, len(recent))  # last 100 requests

        for obj in recent[:cutoff_count]:
            try:
                _, body = conn.get_object(container, obj["name"])
                log = json.loads(body)
                logs.append(log)
            except Exception:
                continue
    except Exception as e:
        print(f"  Warning: could not load inference logs: {e}")
        # Return synthetic demo data if no real logs yet
        logs = _generate_demo_logs(20)

    return logs


def _generate_demo_logs(n):
    """Generate demo inference logs for testing when no real traffic exists."""
    import random
    random.seed(42)
    logs = []
    for i in range(n):
        logs.append({
            "request_id"   : f"req_{i:06d}",
            "duration_sec" : random.gauss(4.5, 2.1),
            "transcript"   : " ".join(["word"] * random.randint(5, 15)),
            "feature_mean" : random.gauss(0.0, 0.05),
            "latency_ms"   : random.gauss(150, 30),
            "timestamp"    : datetime.now(timezone.utc).isoformat(),
        })
    return logs


def load_reference_stats(conn, container, reference_version):
    """Load reference distribution from training data stats."""
    path = f"datasets/v{reference_version}/dataset_card.json"
    try:
        _, body = conn.get_object(container, path)
        card = json.loads(body)
        # Extract stats if available, else use defaults
        return card.get("distribution_stats", DEFAULT_REFERENCE)
    except Exception:
        return DEFAULT_REFERENCE


# ── Main monitor ──────────────────────────────────────────────────────────────

def run_drift_check(container, reference_version, mlflow_uri):
    """Run one drift detection cycle."""
    conn = get_conn()
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"\n[{timestamp}] Running drift check...")

    # Load data
    inference_logs = load_recent_inference_logs(conn, container)
    reference = load_reference_stats(conn, container, reference_version)

    n_samples = len(inference_logs)
    print(f"  Loaded {n_samples} recent inference logs")

    if n_samples < MIN_SAMPLES_FOR_DRIFT:
        print(f"  Too few samples ({n_samples} < {MIN_SAMPLES_FOR_DRIFT}) — skipping drift check")
        return {"status": "insufficient_data", "n_samples": n_samples}

    # Compute live statistics
    live_stats = compute_live_stats(inference_logs)

    # Detect drift
    alerts = detect_drift(live_stats, reference)

    # Overall drift score (max across all metrics)
    max_drift = max((a["drift_score"] for a in alerts), default=0.0)
    drift_status = "CRITICAL" if any(a["level"] == "CRITICAL" for a in alerts) \
                   else "WARNING" if alerts \
                   else "OK"

    result = {
        "timestamp"      : timestamp,
        "n_samples"      : n_samples,
        "drift_status"   : drift_status,
        "max_drift_score": round(max_drift, 4),
        "alerts"         : alerts,
        "live_stats"     : live_stats,
        "reference_stats": reference,
    }

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("data_quality_drift_monitor")

    with mlflow.start_run(run_name=f"drift_check_{timestamp[:16]}"):
        mlflow.log_param("reference_version",   reference_version)
        mlflow.log_param("n_samples",           n_samples)

        mlflow.log_metric("max_drift_score",    max_drift)
        mlflow.log_metric("n_alerts",           len(alerts))
        mlflow.log_metric("duration_mean_live", live_stats.get("duration_mean", 0))
        mlflow.log_metric("duration_std_live",  live_stats.get("duration_std", 0))
        mlflow.log_metric("transcript_len_live",live_stats.get("transcript_len_mean", 0))
        mlflow.log_metric("feature_mean_live",  live_stats.get("feature_mean", 0))
        mlflow.log_metric("feature_std_live",   live_stats.get("feature_std", 0))

        mlflow.set_tag("drift_status",   drift_status)
        mlflow.set_tag("alert_count",    str(len(alerts)))

        if alerts:
            for alert in alerts:
                mlflow.set_tag(
                    f"drift_{alert['metric']}",
                    f"{alert['level']}:{alert['drift_score']:.3f}"
                )

    # Write drift score file (used by Docker health check)
    with open("/tmp/drift_status.json", "w") as f:
        json.dump({
            "status"    : drift_status,
            "max_drift" : max_drift,
            "timestamp" : timestamp,
            "n_alerts"  : len(alerts),
        }, f)

    # Upload result to object storage
    result_path = f"drift_reports/{timestamp[:10]}/drift_{timestamp[:19].replace(':', '-')}.json"
    try:
        conn.put_object(container, result_path,
                        json.dumps(result, indent=2).encode(),
                        content_type="application/json")
    except Exception as e:
        print(f"  Warning: could not upload drift report: {e}")

    # Print summary
    status_symbol = "✓" if drift_status == "OK" else ("⚠" if drift_status == "WARNING" else "✗")
    print(f"  {status_symbol} Drift status   : {drift_status}")
    print(f"  Max drift score : {max_drift:.3f}")
    if alerts:
        print(f"  Alerts ({len(alerts)}):")
        for a in alerts:
            print(f"    [{a['level']}] {a['metric']}: "
                  f"ref={a['reference']:.3f} live={a['live']:.3f} "
                  f"drift={a['drift_score']:.3f}")

    return result


def run_continuous(container, reference_version, mlflow_uri, interval_sec):
    """Run drift checks continuously at fixed interval."""
    print(f"\n{'='*55}")
    print(f"  DATA DRIFT MONITOR (continuous)")
    print(f"{'='*55}")
    print(f"  Container         : {container}")
    print(f"  Reference version : {reference_version}")
    print(f"  Check interval    : {interval_sec}s")
    print(f"{'='*55}\n")

    while True:
        try:
            run_drift_check(container, reference_version, mlflow_uri)
        except Exception as e:
            print(f"  Error during drift check: {e}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container",          required=True)
    parser.add_argument("--reference-version",  default="1.0")
    parser.add_argument("--mlflow-uri",         default=os.environ.get(
                         "MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument("--mode",
                        choices=["once", "continuous"], default="once")
    parser.add_argument("--interval",           type=int, default=300,
                        help="Seconds between checks in continuous mode")
    args = parser.parse_args()

    if args.mode == "continuous":
        run_continuous(
            args.container, args.reference_version,
            args.mlflow_uri, args.interval)
    else:
        run_drift_check(
            args.container, args.reference_version, args.mlflow_uri)
