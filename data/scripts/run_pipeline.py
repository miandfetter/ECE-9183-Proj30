#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Runs the full data pipeline end-to-end:
  Stage 1: Ingest from HuggingFace → Object Storage
  Stage 2: Preprocess (resample, normalize, split)
  Stage 3: Synthetic data generation (augmentation)

Usage:
    python3 run_pipeline.py --container ObjStore_projecttranscriptionmirotalk

Requirements:
    source openrc  # before running
"""

import argparse
import subprocess
import sys
import time

STAGES = [
    ("Ingestion",            "01_ingest.py"),
    ("Preprocessing",        "02_preprocess.py"),
    ("Synthetic Generation", "03_synthetic_data.py"),
]


def run_stage(name, script, extra_args):
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script] + extra_args,
        check=False
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ Stage '{name}' failed (exit code {result.returncode})")
        sys.exit(1)
    print(f"\n✓ Stage '{name}' completed in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--split",           default="test")
    parser.add_argument("--limit",           type=int, default=20)
    parser.add_argument("--dataset-version", default="1.0")
    parser.add_argument("--multiplier",      type=int, default=3)
    parser.add_argument("--stage",           type=int, default=0,
                        help="Start from stage N (1=ingest, 2=preprocess, 3=synthetic)")
    args = parser.parse_args()

    common         = ["--container", args.container, "--dataset-version", args.dataset_version]
    ingest_args    = ["--container", args.container, "--split", args.split, "--limit", str(args.limit)]
    stage_args = [
        ingest_args,
        common,
        ["--multiplier", str(args.multiplier)] + common,
    ]

    for idx, (name, script) in enumerate(STAGES):
        if idx + 1 < args.stage:
            print(f"Skipping stage {idx+1}: {name}")
            continue
        run_stage(name, script, stage_args[idx])

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Container : {args.container}")
    print(f"Version   : {args.dataset_version}")
    print(f"\nFolder structure in object storage:")
    print(f"  raw/v{args.dataset_version}/                  ← ingested data")
    print(f"  processed/v{args.dataset_version}/            ← preprocessed + splits")
    print(f"  synthetic/v{args.dataset_version}/            ← augmented training data")
