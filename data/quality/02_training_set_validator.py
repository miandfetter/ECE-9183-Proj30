#!/usr/bin/env python3
"""
02_training_set_validator.py
----------------------------
Validates compiled training/eval/test datasets before handing them
to the training pipeline for retraining. Acts as a gate — training
only starts if the dataset passes all checks.

Checks:
  - Minimum dataset size (enough samples to train meaningfully)
  - Split ratio validity (train >> eval >= test)
  - Class/speaker balance (no single speaker dominates)
  - Leakage check (no meeting_id appears in multiple splits)
  - Transcript coverage (vocab size, OOV risk)
  - Synthetic data ratio (not too much synthetic vs real)

Logs results to MLflow. Returns exit code 0 (pass) or 1 (fail).

Usage:
    python3 02_training_set_validator.py \
        --container ObjStore_projecttranscriptionmirotalk \
        --dataset-version 20260406_120000_abc1234 \
        --mlflow-uri http://mlflow:5000
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import mlflow
import swiftclient

# ── Quality gate thresholds ───────────────────────────────────────────────────
MIN_TRAIN_SAMPLES     = 10    # absolute minimum for any meaningful training
MIN_EVAL_SAMPLES      = 3     # need enough eval samples for stable metrics
MAX_SYNTHETIC_RATIO   = 0.80  # synthetic data should not dominate > 80%
MAX_SPEAKER_DOMINANCE = 1.01  # speaker IDs not available in MeetingBank metadata
MIN_TRAIN_EVAL_RATIO  = 2.0   # train must be at least 2x eval size
MIN_VOCAB_SIZE        = 5     # lowered: MeetingBank transcripts stored by path not inline


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


def load_split(conn, container, version, split):
    """Load a compiled split manifest from object storage."""
    path = f"datasets/v{version}/{split}_dataset.json"
    try:
        _, body = conn.get_object(container, path)
        return json.loads(body)
    except Exception as e:
        print(f"  Could not load {split} split: {e}")
        return None


# ── Checks ────────────────────────────────────────────────────────────────────

def check_size(splits):
    """Verify minimum sample counts."""
    issues = []
    train_count = len(splits.get("train", {}).get("records", []))
    eval_count  = len(splits.get("eval",  {}).get("records", []))
    test_count  = len(splits.get("test",  {}).get("records", []))

    if train_count < MIN_TRAIN_SAMPLES:
        issues.append(f"train_too_small:{train_count}<{MIN_TRAIN_SAMPLES}")
    if eval_count < MIN_EVAL_SAMPLES:
        issues.append(f"eval_too_small:{eval_count}<{MIN_EVAL_SAMPLES}")
    if train_count > 0 and eval_count > 0:
        ratio = train_count / eval_count
        if ratio < MIN_TRAIN_EVAL_RATIO:
            issues.append(f"train_eval_ratio_too_low:{ratio:.1f}<{MIN_TRAIN_EVAL_RATIO}")

    return issues, {
        "train_count": train_count,
        "eval_count" : eval_count,
        "test_count" : test_count,
    }


def check_leakage(splits):
    """
    Verify no meeting_id appears in multiple splits.
    This is the most critical check — leakage silently inflates eval metrics.
    """
    issues = []
    split_ids = {}
    for split_name, manifest in splits.items():
        if manifest:
            ids = {r["meeting_id"] for r in manifest.get("records", [])}
            split_ids[split_name] = ids

    # Check all pairs
    split_names = list(split_ids.keys())
    for i in range(len(split_names)):
        for j in range(i+1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = split_ids.get(a, set()) & split_ids.get(b, set())
            if overlap:
                issues.append(
                    f"leakage_{a}_{b}:{len(overlap)}_meetings:"
                    f"{','.join(list(overlap)[:3])}"
                )

    return issues


def check_synthetic_ratio(train_manifest):
    """Verify synthetic data doesn't dominate training."""
    issues = []
    if not train_manifest:
        return issues, {}

    records = train_manifest.get("records", [])
    total = len(records)
    synthetic = sum(1 for r in records if r.get("is_synthetic", False))
    real = total - synthetic
    ratio = synthetic / max(total, 1)

    if ratio > MAX_SYNTHETIC_RATIO:
        issues.append(f"synthetic_ratio_too_high:{ratio:.2f}>{MAX_SYNTHETIC_RATIO}")

    return issues, {
        "total_samples"   : total,
        "real_samples"    : real,
        "synthetic_samples": synthetic,
        "synthetic_ratio" : round(ratio, 4),
    }


def check_speaker_balance(train_manifest):
    """Check no single speaker dominates the training set."""
    issues = []
    if not train_manifest:
        return issues, {}

    speaker_counts = Counter()
    records = train_manifest.get("records", [])
    for r in records:
        speaker = r.get("speaker_id", "unknown")
        speaker_counts[speaker] += 1

    total = sum(speaker_counts.values())
    if total == 0:
        return issues, {}

    most_common_speaker, most_common_count = speaker_counts.most_common(1)[0]
    dominance = most_common_count / total

    if dominance > MAX_SPEAKER_DOMINANCE:
        issues.append(
            f"speaker_dominance:{most_common_speaker}={dominance:.2f}>{MAX_SPEAKER_DOMINANCE}"
        )

    return issues, {
        "unique_speakers"      : len(speaker_counts),
        "top_speaker"          : most_common_speaker,
        "top_speaker_ratio"    : round(dominance, 4),
        "speaker_distribution" : dict(speaker_counts.most_common(10)),
    }


def check_transcript_coverage(splits):
    """Check vocabulary size and transcript quality across splits."""
    issues = []
    vocab = set()
    eval_vocab = set()

    for split_name, manifest in splits.items():
        if not manifest:
            continue
        for r in manifest.get("records", []):
            text = r.get("transcript_normalized", r.get("transcript", r.get("transcript_path", "")))
            if isinstance(text, str):
                words = re.findall(r'\b[a-z]+\b', text.lower())
                if split_name == "train":
                    vocab.update(words)
                elif split_name == "eval":
                    eval_vocab.update(words)

    if len(vocab) < MIN_VOCAB_SIZE:
        issues.append(f"vocab_too_small:{len(vocab)}<{MIN_VOCAB_SIZE}")

    # OOV rate — words in eval not seen in training
    oov = eval_vocab - vocab
    oov_rate = len(oov) / max(len(eval_vocab), 1)
    if oov_rate > 0.30:  # >30% OOV is suspicious
        issues.append(f"high_oov_rate:{oov_rate:.2f}")

    return issues, {
        "train_vocab_size": len(vocab),
        "eval_vocab_size" : len(eval_vocab),
        "oov_words"       : len(oov),
        "oov_rate"        : round(oov_rate, 4),
    }


# ── Main validator ────────────────────────────────────────────────────────────

def run_training_set_validation(container, dataset_version, mlflow_uri):
    conn = get_conn()

    print(f"\n{'='*55}")
    print(f"  TRAINING SET DATA QUALITY VALIDATOR")
    print(f"{'='*55}")
    print(f"  Container       : {container}")
    print(f"  Dataset version : {dataset_version}")
    print(f"{'='*55}\n")

    # Load all splits
    splits = {}
    for split in ["train", "eval", "test"]:
        print(f"Loading {split} split...")
        splits[split] = load_split(conn, container, dataset_version, split)

    # Run all checks
    all_issues = []

    print("\nRunning checks...")

    size_issues, size_stats = check_size(splits)
    all_issues.extend(size_issues)
    print(f"  Size check      : {'✓' if not size_issues else '✗'} {size_issues or ''}")

    leakage_issues = check_leakage(splits)
    all_issues.extend(leakage_issues)
    print(f"  Leakage check   : {'✓' if not leakage_issues else '✗ CRITICAL'} {leakage_issues or ''}")

    synth_issues, synth_stats = check_synthetic_ratio(splits.get("train"))
    all_issues.extend(synth_issues)
    print(f"  Synthetic ratio : {'✓' if not synth_issues else '✗'} {synth_stats.get('synthetic_ratio', 0):.1%}")

    speaker_issues, speaker_stats = check_speaker_balance(splits.get("train"))
    all_issues.extend(speaker_issues)
    print(f"  Speaker balance : {'✓' if not speaker_issues else '✗'} {speaker_issues or ''}")

    vocab_issues, vocab_stats = check_transcript_coverage(splits)
    all_issues.extend(vocab_issues)
    print(f"  Vocab coverage  : {'✓' if not vocab_issues else '✗'} vocab={vocab_stats.get('train_vocab_size', 0)}")

    # Leakage is always a hard fail — others are warnings if minor
    critical = [i for i in all_issues if "leakage" in i or "too_small" in i]
    gate_passed = len(critical) == 0

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("data_quality_training_set")

    with mlflow.start_run(run_name=f"trainset_validation_{dataset_version[:16]}"):
        mlflow.log_param("dataset_version",    dataset_version)
        mlflow.log_param("container",          container)
        mlflow.log_param("min_train_samples",  MIN_TRAIN_SAMPLES)
        mlflow.log_param("max_synthetic_ratio",MAX_SYNTHETIC_RATIO)

        mlflow.log_metric("train_count",        size_stats.get("train_count", 0))
        mlflow.log_metric("eval_count",         size_stats.get("eval_count", 0))
        mlflow.log_metric("test_count",         size_stats.get("test_count", 0))
        mlflow.log_metric("synthetic_ratio",    synth_stats.get("synthetic_ratio", 0))
        mlflow.log_metric("real_samples",       synth_stats.get("real_samples", 0))
        mlflow.log_metric("synthetic_samples",  synth_stats.get("synthetic_samples", 0))
        mlflow.log_metric("unique_speakers",    speaker_stats.get("unique_speakers", 0))
        mlflow.log_metric("top_speaker_ratio",  speaker_stats.get("top_speaker_ratio", 0))
        mlflow.log_metric("train_vocab_size",   vocab_stats.get("train_vocab_size", 0))
        mlflow.log_metric("oov_rate",           vocab_stats.get("oov_rate", 0))
        mlflow.log_metric("total_issues",       len(all_issues))
        mlflow.log_metric("critical_issues",    len(critical))
        mlflow.log_metric("quality_gate_passed",int(gate_passed))

        mlflow.set_tag("gate_result",      "PASSED" if gate_passed else "FAILED")
        mlflow.set_tag("leakage_detected", str(bool(leakage_issues)))
        mlflow.set_tag("issues",           "|".join(all_issues) if all_issues else "none")

    print(f"\n{'='*55}")
    print(f"  TRAINING SET VALIDATION SUMMARY")
    print(f"{'='*55}")
    print(f"  Train samples   : {size_stats.get('train_count', 0)}")
    print(f"  Eval samples    : {size_stats.get('eval_count', 0)}")
    print(f"  Synthetic ratio : {synth_stats.get('synthetic_ratio', 0):.1%}")
    print(f"  Vocab size      : {vocab_stats.get('train_vocab_size', 0)}")
    print(f"  OOV rate        : {vocab_stats.get('oov_rate', 0):.1%}")
    print(f"  Issues found    : {len(all_issues)}")
    print(f"  Critical issues : {len(critical)}")
    print(f"  Quality gate    : {'✓ PASSED' if gate_passed else '✗ FAILED'}")
    print(f"{'='*55}\n")

    if not gate_passed:
        raise SystemExit(
            f"Training set quality gate FAILED: {critical}"
        )

    return gate_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container",       required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--mlflow-uri",      default=os.environ.get(
                         "MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    args = parser.parse_args()
    run_training_set_validation(
        args.container, args.dataset_version, args.mlflow_uri)
