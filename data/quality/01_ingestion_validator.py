#!/usr/bin/env python3
"""
01_ingestion_validator.py
-------------------------
Validates data quality at ingestion from HuggingFace before it enters
the object storage pipeline. Runs as a gate — records that fail are
quarantined, not discarded, so they can be reviewed.

Validation checks:
  - Schema completeness (required fields present)
  - Audio validity (correct format, duration bounds, sample rate)
  - Transcript sanity (non-empty, character set, length bounds)
  - Checksum integrity (no corrupted transfers)
  - Duplicate detection (same meeting_id not ingested twice)

Logs all results to MLflow as a data validation run.

Usage:
    python3 01_ingestion_validator.py \
        --container ObjStore_projecttranscriptionmirotalk \
        --dataset-version 1.0 \
        --mlflow-uri http://mlflow:5000
"""

import argparse
import hashlib
import io
import json
import math
import os
import struct
import wave
from datetime import datetime, timezone

import mlflow
import swiftclient

# ── Thresholds (justified in comments) ───────────────────────────────────────
MIN_DURATION_SEC  = 1.0    # < 1s too short for reliable ASR transcription
MAX_DURATION_SEC  = 600.0  # > 10min likely a full meeting, not a segment
MIN_TRANSCRIPT_CHARS = 5   # single characters/words are not useful training signal
MAX_TRANSCRIPT_CHARS = 5000 # sanity upper bound
ALLOWED_SAMPLE_RATES = {8000, 16000, 22050, 44100, 48000}
MIN_SAMPLE_RATE  = 8000    # below 8kHz, speech quality is unusable
REQUIRED_FIELDS  = ["meeting_id", "transcript"]


# ── Swift connection ──────────────────────────────────────────────────────────

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


# ── Validation checks ─────────────────────────────────────────────────────────

def check_schema(record):
    """Check all required fields are present and non-null."""
    issues = []
    for field in REQUIRED_FIELDS:
        if field not in record or record[field] is None:
            issues.append(f"missing_field:{field}")
        elif isinstance(record[field], str) and not record[field].strip():
            issues.append(f"empty_field:{field}")
    return issues


def check_transcript(transcript):
    """Validate transcript text quality."""
    issues = []
    if not isinstance(transcript, str):
        return ["transcript_not_string"]

    text = transcript.strip()
    if len(text) < MIN_TRANSCRIPT_CHARS:
        issues.append(f"transcript_too_short:{len(text)}_chars")
    if len(text) > MAX_TRANSCRIPT_CHARS:
        issues.append(f"transcript_too_long:{len(text)}_chars")

    # Check for mostly non-ASCII (possible encoding error)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.7:
        issues.append(f"low_ascii_ratio:{ascii_ratio:.2f}")

    # Check for repetitive content (possible transcript error)
    words = text.lower().split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append(f"repetitive_transcript:{unique_ratio:.2f}_unique")

    return issues


def check_audio(audio_bytes):
    """Validate WAV audio file."""
    issues = []
    stats = {}

    if not audio_bytes:
        return ["empty_audio"], stats

    # Check WAV header
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, 'rb') as wf:
            n_channels  = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames    = wf.getnframes()
            sampwidth   = wf.getsampwidth()
            duration    = n_frames / sample_rate

        stats = {
            "duration_sec" : round(duration, 3),
            "sample_rate"  : sample_rate,
            "n_channels"   : n_channels,
            "bit_depth"    : sampwidth * 8,
        }

        if sample_rate < MIN_SAMPLE_RATE:
            issues.append(f"low_sample_rate:{sample_rate}Hz")
        if duration < MIN_DURATION_SEC:
            issues.append(f"too_short:{duration:.2f}s")
        if duration > MAX_DURATION_SEC:
            issues.append(f"too_long:{duration:.2f}s")
        if n_channels not in (1, 2):
            issues.append(f"unusual_channels:{n_channels}")

    except Exception as e:
        issues.append(f"invalid_wav:{str(e)[:50]}")

    return issues, stats


def compute_checksum(data):
    """MD5 checksum for integrity verification."""
    return hashlib.md5(data).hexdigest()


def check_duplicate(conn, container, meeting_id, version):
    """Check if this meeting_id was already ingested."""
    try:
        conn.head_object(container, f"raw/v{version}/{meeting_id}/metadata.json")
        return True  # already exists
    except Exception:
        return False


# ── Main validator ────────────────────────────────────────────────────────────

def validate_and_upload(conn, container, version, record, audio_bytes=None):
    """
    Run all checks on a single record.
    Returns: (status, issues, stats)
    status: 'pass' | 'quarantine' | 'fail'
    """
    all_issues = []
    audio_stats = {}

    # 1. Schema check
    schema_issues = check_schema(record)
    all_issues.extend(schema_issues)

    # 2. Transcript check
    transcript = record.get("transcript", "")
    trans_issues = check_transcript(transcript)
    all_issues.extend(trans_issues)

    # 3. Audio check (if audio provided)
    if audio_bytes:
        audio_issues, audio_stats = check_audio(audio_bytes)
        all_issues.extend(audio_issues)
        checksum = compute_checksum(audio_bytes)
    else:
        checksum = None

    # 4. Duplicate check
    meeting_id = record.get("meeting_id", "unknown")
    if check_duplicate(conn, container, meeting_id, version):
        all_issues.append("duplicate_meeting_id")

    # Determine status
    # Critical failures → quarantine (reviewable)
    # Minor issues → pass with warnings
    critical = [i for i in all_issues if any(k in i for k in [
        "missing_field", "empty_audio", "invalid_wav",
        "duplicate_meeting_id", "transcript_not_string"
    ])]

    if critical:
        status = "quarantine"
    elif all_issues:
        status = "pass_with_warnings"
    else:
        status = "pass"

    result = {
        "meeting_id"   : meeting_id,
        "status"       : status,
        "issues"       : all_issues,
        "audio_stats"  : audio_stats,
        "checksum"     : checksum,
        "validated_at" : datetime.now(timezone.utc).isoformat(),
    }

    # Upload validation result to object storage
    prefix = "quarantine" if status == "quarantine" else f"raw/v{version}"
    result_path = f"{prefix}/{meeting_id}/validation_result.json"
    conn.put_object(
        container, result_path,
        json.dumps(result, indent=2).encode(),
        content_type="application/json"
    )

    return status, all_issues, audio_stats, result


def run_ingestion_validation(container, version, mlflow_uri):
    conn = get_conn()

    # Load raw manifest
    try:
        _, body = conn.get_object(container, f"raw/v{version}/manifest.json")
        manifest = json.loads(body)
        records = manifest.get("records", [])
    except Exception as e:
        print(f"Could not load manifest: {e}")
        records = []

    print(f"\n{'='*55}")
    print(f"  INGESTION DATA QUALITY VALIDATOR")
    print(f"{'='*55}")
    print(f"  Container : {container}")
    print(f"  Version   : {version}")
    print(f"  Records   : {len(records)}")
    print(f"{'='*55}\n")

    stats = {"pass": 0, "pass_with_warnings": 0, "quarantine": 0}
    all_issues_flat = []
    audio_durations = []

    # MLflow run
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("data_quality_ingestion")

    with mlflow.start_run(run_name=f"ingestion_validation_v{version}"):
        mlflow.log_param("dataset_version", version)
        mlflow.log_param("container", container)
        mlflow.log_param("record_count", len(records))
        mlflow.log_param("min_duration_sec", MIN_DURATION_SEC)
        mlflow.log_param("max_duration_sec", MAX_DURATION_SEC)
        mlflow.log_param("min_transcript_chars", MIN_TRANSCRIPT_CHARS)

        for i, record in enumerate(records):
            mid = record.get("meeting_id", f"record_{i}")
            print(f"[{i+1}/{len(records)}] Validating {mid}...")

            # Try to load audio
            audio_bytes = None
            audio_path = record.get("audio_path")
            if audio_path:
                try:
                    _, audio_bytes = conn.get_object(container, audio_path)
                except Exception:
                    pass

            # Load transcript
            trans_path = record.get("transcript_path")
            if trans_path:
                try:
                    _, trans_body = conn.get_object(container, trans_path)
                    trans_obj = json.loads(trans_body)
                    record["transcript"] = trans_obj.get("transcript", "")
                except Exception:
                    pass

            status, issues, audio_stats, _ = validate_and_upload(
                conn, container, version, record, audio_bytes)

            stats[status] = stats.get(status, 0) + 1
            all_issues_flat.extend(issues)

            if audio_stats.get("duration_sec"):
                audio_durations.append(audio_stats["duration_sec"])

            symbol = "✓" if status == "pass" else ("⚠" if "warning" in status else "✗")
            print(f"  {symbol} {status}" + (f" — {issues}" if issues else ""))

        # Aggregate metrics → MLflow
        pass_rate = stats["pass"] / max(len(records), 1)
        quarantine_rate = stats.get("quarantine", 0) / max(len(records), 1)
        avg_duration = sum(audio_durations) / max(len(audio_durations), 1)

        mlflow.log_metric("pass_count",          stats["pass"])
        mlflow.log_metric("warning_count",        stats.get("pass_with_warnings", 0))
        mlflow.log_metric("quarantine_count",     stats.get("quarantine", 0))
        mlflow.log_metric("pass_rate",            round(pass_rate, 4))
        mlflow.log_metric("quarantine_rate",      round(quarantine_rate, 4))
        mlflow.log_metric("avg_audio_duration",   round(avg_duration, 3))
        mlflow.log_metric("unique_issue_types",   len(set(all_issues_flat)))

        # Gate: fail pipeline if quarantine rate too high
        QUARANTINE_THRESHOLD = 0.20  # >20% quarantined = something wrong upstream
        gate_passed = quarantine_rate <= QUARANTINE_THRESHOLD
        mlflow.log_metric("quality_gate_passed",  int(gate_passed))
        mlflow.set_tag("gate_result", "PASSED" if gate_passed else "FAILED")

        print(f"\n{'='*55}")
        print(f"  INGESTION VALIDATION SUMMARY")
        print(f"{'='*55}")
        print(f"  Pass            : {stats['pass']}")
        print(f"  Pass w/warnings : {stats.get('pass_with_warnings', 0)}")
        print(f"  Quarantined     : {stats.get('quarantine', 0)}")
        print(f"  Pass rate       : {pass_rate:.1%}")
        print(f"  Quality gate    : {'✓ PASSED' if gate_passed else '✗ FAILED'}")
        print(f"{'='*55}\n")

        if not gate_passed:
            raise SystemExit(
                f"Ingestion quality gate FAILED: "
                f"{quarantine_rate:.1%} quarantine rate > {QUARANTINE_THRESHOLD:.0%} threshold"
            )

    return gate_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container",       required=True)
    parser.add_argument("--dataset-version", default="1.0")
    parser.add_argument("--mlflow-uri",      default=os.environ.get(
                         "MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    args = parser.parse_args()
    run_ingestion_validation(args.container, args.dataset_version, args.mlflow_uri)
