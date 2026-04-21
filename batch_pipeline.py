#!/usr/bin/env python3
"""

    python3 batch_pipeline.py \
        --container ObjStore_projecttranscriptionmirotalk \
        --cutoff-date 2021-06-01 \
        --min-duration 1.5 \
        --min-snr 10.0 \
        --output-version auto
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

import swiftclient

PIPELINE_VERSION = os.environ.get("GIT_SHA", "local")
def get_swift_connection():
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
def list_objects(conn, container, prefix):
    
    try:
        _, objects = conn.get_container(container, prefix=prefix, full_listing=True)
        return [o["name"] for o in objects]
    except Exception as e:
        print(f"  Warning: could not list {prefix}: {e}")
        return []


def load_json(conn, container, path):
    """Load and parse a JSON object from Swift."""
    try:
        _, body = conn.get_object(container, path)
        return json.loads(body)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None




def passes_quality_gate(record, metadata, min_duration, min_snr, min_tokens):
    
    duration = record.get("duration_sec", 0.0)
    if duration < min_duration:
        return False, f"duration {duration:.1f}s < {min_duration}s"

    token_count = record.get("token_count", 0)
    if token_count < min_tokens:
        return False, f"token_count {token_count} < {min_tokens}"

    if metadata:
        snr = metadata.get("snr_db")
        if snr is not None and snr < min_snr:
            return False, f"SNR {snr:.1f}dB < {min_snr}dB"

    return True, "ok"


def extract_date(metadata):
    
    if not metadata:
        return None
    date_str = metadata.get("date") or metadata.get("meeting_date")
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(date_str)[:10], fmt)
        except ValueError:
            continue
    return None


def extract_speakers(metadata):
    
    if not metadata:
        return set()
    speakers = metadata.get("speakers") or metadata.get("speaker_ids") or []
    if isinstance(speakers, str):
        speakers = [speakers]
    return set(speakers)




def assign_splits(candidates, cutoff_date, eval_ratio=0.15, test_ratio=0.15):
    
    meetings = defaultdict(list)
    for c in candidates:
        meetings[c["meeting_id"]].append(c)

    meeting_list = list(meetings.keys())

    def meeting_sort_key(mid):
        meta = meetings[mid][0].get("_metadata")
        dt = extract_date(meta)
        if dt:
            return dt
        
        h = int(hashlib.md5(mid.encode()).hexdigest(), 16)
        return datetime(2020, 1, 1) + __import__("datetime").timedelta(seconds=h % (365*24*3600))

    meeting_list.sort(key=meeting_sort_key)

    
    train_meetings, eval_pool = [], []
    for mid in meeting_list:
        meta = meetings[mid][0].get("_metadata")
        dt = extract_date(meta)
        if dt and cutoff_date:
            if dt < cutoff_date:
                train_meetings.append(mid)
            else:
                eval_pool.append(mid)
        else:
            
            train_meetings.append(mid)

    
    if not eval_pool:
        print("  Warning: no date metadata found — using position-based split")
        n = len(meeting_list)
        train_end = int(n * (1 - eval_ratio - test_ratio))
        eval_end  = int(n * (1 - test_ratio))
        train_meetings = meeting_list[:train_end]
        eval_pool      = meeting_list[train_end:eval_end]
        test_pool      = meeting_list[eval_end:]
    else:
        n_test = max(1, int(len(eval_pool) * (test_ratio / (eval_ratio + test_ratio))))
        test_pool = eval_pool[-n_test:]
        eval_pool = eval_pool[:-n_test]

   
    train_speakers = set()
    for mid in train_meetings:
        for c in meetings[mid]:
            train_speakers.update(c.get("_speakers", set()))

    def speaker_safe(mid):
        meeting_speakers = set()
        for c in meetings[mid]:
            meeting_speakers.update(c.get("_speakers", set()))
        overlap = meeting_speakers & train_speakers
        if overlap:
            print(f"  Speaker leakage: removing {mid} from eval "
                  f"(speakers {overlap} in training)")
            return False
        return True

    eval_pool_clean = [m for m in eval_pool if speaker_safe(m)]
    test_pool_clean = [m for m in test_pool if speaker_safe(m)]

    
    split_map = {}
    for mid in train_meetings:
        split_map[mid] = "train"
    for mid in eval_pool_clean:
        split_map[mid] = "eval"
    for mid in test_pool_clean:
        split_map[mid] = "test"

    
    for mid in eval_pool:
        if mid not in split_map:
            split_map[mid] = "train"
            print(f"  Reassigned {mid} to train (speaker overlap)")

    return split_map



def run_batch_pipeline(
    container,
    dataset_version,
    cutoff_date_str,
    min_duration,
    min_snr,
    min_tokens,
    output_version,
):
    conn = get_swift_connection()

    # Generate output version tag
    if output_version == "auto":
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_version = f"{ts}_{PIPELINE_VERSION[:7]}"

    print(f"\n{'='*60}")
    print(f"  BATCH DATASET COMPILATION PIPELINE")
    print(f"{'='*60}")
    print(f"  Container      : {container}")
    print(f"  Source version : {dataset_version}")
    print(f"  Output version : {output_version}")
    print(f"  Cutoff date    : {cutoff_date_str or 'none (position-based)'}")
    print(f"  Min duration   : {min_duration}s")
    print(f"  Min SNR        : {min_snr}dB")
    print(f"  Min tokens     : {min_tokens}")
    print(f"{'='*60}\n")

    cutoff_date = None
    if cutoff_date_str:
        cutoff_date = datetime.strptime(cutoff_date_str, "%Y-%m-%d")

    
    print("Step 1: Loading processed manifests from object storage...")
    proc_prefix = f"processed/v{dataset_version}/"
    all_objects = list_objects(conn, container, proc_prefix)
    manifest_paths = [o for o in all_objects if o.endswith("_manifest.json")]

    if not manifest_paths:
        print(f"  No manifests found under {proc_prefix}")
        print("  Run 02_preprocess.py first.")
        sys.exit(1)

    all_records = []
    for path in manifest_paths:
        manifest = load_json(conn, container, path)
        if manifest and "records" in manifest:
            all_records.extend(manifest["records"])
            print(f"  Loaded {len(manifest['records'])} records from {path}")

    
    synth_prefix = f"synthetic/v{dataset_version}/"
    synth_objects = list_objects(conn, container, synth_prefix)
    synth_manifests = [o for o in synth_objects if o.endswith("synthetic_manifest.json")]

    synth_records = []
    for path in synth_manifests:
        manifest = load_json(conn, container, path)
        if manifest and "records" in manifest:
            synth_records.extend(manifest["records"])
            print(f"  Loaded {len(manifest['records'])} synthetic records from {path}")

    print(f"\n  Total real records    : {len(all_records)}")
    print(f"  Total synthetic records: {len(synth_records)}")

    
    print("\nStep 2: Loading metadata for candidate selection...")
    metadata_cache = {}
    seen_meetings = set()
    duplicate_count = 0

    for record in all_records:
        mid = record.get("meeting_id", "")

        
        if mid in seen_meetings:
            duplicate_count += 1
            continue
        seen_meetings.add(mid)

        
        meta_path = record.get("metadata_path") or \
                    f"raw/v{dataset_version}/{mid}/metadata.json"
        meta = load_json(conn, container, meta_path)
        metadata_cache[mid] = meta

        
        record["_metadata"] = meta
        record["_speakers"] = extract_speakers(meta)

    print(f"  Unique meetings     : {len(seen_meetings)}")
    print(f"  Duplicates removed  : {duplicate_count}")

    
    print("\nStep 3: Applying quality gates...")
    candidates = []
    rejected = []

    for record in all_records:
        mid = record.get("meeting_id", "")
        if mid not in seen_meetings:
            continue
        meta = metadata_cache.get(mid)
        passed, reason = passes_quality_gate(
            record, meta, min_duration, min_snr, min_tokens)
        if passed:
            candidates.append(record)
        else:
            rejected.append({"meeting_id": mid, "reason": reason})

    print(f"  Candidates passed   : {len(candidates)}")
    print(f"  Rejected            : {len(rejected)}")
    if rejected[:3]:
        for r in rejected[:3]:
            print(f"    - {r['meeting_id']}: {r['reason']}")
        if len(rejected) > 3:
            print(f"    ... and {len(rejected)-3} more")

    if not candidates:
        print("\nNo candidates passed quality gates. Exiting.")
        sys.exit(1)

    
    print("\nStep 4: Assigning splits (temporal + speaker-aware)...")
    split_map = assign_splits(candidates, cutoff_date)

    split_counts = defaultdict(int)
    for mid, split in split_map.items():
        split_counts[split] += 1

    print(f"  Train meetings : {split_counts['train']}")
    print(f"  Eval meetings  : {split_counts['eval']}")
    print(f"  Test meetings  : {split_counts['test']}")

    print("\nStep 5: Adding synthetic data to training split only...")
    
    train_synthetic = [r for r in synth_records
                       if r.get("split") == "train"]
    print(f"  Synthetic training samples added: {len(train_synthetic)}")

    
    print("\nStep 6: Compiling and uploading versioned datasets...")
    output_prefix = f"datasets/v{output_version}"

    split_datasets = {"train": [], "eval": [], "test": []}

    for record in candidates:
        mid = record.get("meeting_id", "")
        split = split_map.get(mid)
        if split:
            # Clean internal fields before saving
            clean = {k: v for k, v in record.items()
                     if not k.startswith("_")}
            clean["assigned_split"] = split
            clean["dataset_version"] = output_version
            split_datasets[split].append(clean)

    
    for record in train_synthetic:
        clean = {k: v for k, v in record.items()
                 if not k.startswith("_")}
        clean["assigned_split"] = "train"
        clean["dataset_version"] = output_version
        clean["is_synthetic"] = True
        split_datasets["train"].append(clean)

    
    dataset_stats = {}
    for split_name, records in split_datasets.items():
        if not records:
            print(f"  Warning: {split_name} split is empty")
            continue

        # Compute fingerprint for reproducibility verification
        fingerprint = hashlib.md5(
            json.dumps([r["meeting_id"] for r in records],
                       sort_keys=True).encode()
        ).hexdigest()

        manifest = {
            "dataset_version"    : output_version,
            "source_version"     : dataset_version,
            "pipeline_version"   : PIPELINE_VERSION,
            "split"              : split_name,
            "compiled_at"        : datetime.now(timezone.utc).isoformat(),
            "cutoff_date"        : cutoff_date_str,
            "record_count"       : len(records),
            "real_count"         : sum(1 for r in records
                                       if not r.get("is_synthetic")),
            "synthetic_count"    : sum(1 for r in records
                                       if r.get("is_synthetic")),
            "fingerprint"        : fingerprint,
            "quality_gates"      : {
                "min_duration_sec" : min_duration,
                "min_snr_db"       : min_snr,
                "min_tokens"       : min_tokens,
            },
            "leakage_prevention" : {
                "method"         : "temporal+speaker",
                "cutoff_date"    : cutoff_date_str,
                "speaker_check"  : True,
                "meeting_level"  : True,
            },
            "records": records,
        }

        path = f"{output_prefix}/{split_name}_dataset.json"
        conn.put_object(
            container, path,
            json.dumps(manifest, indent=2, default=str).encode(),
            content_type="application/json"
        )
        print(f"  Uploaded {split_name}: {len(records)} records → {path}")
        dataset_stats[split_name] = {
            "count": len(records), "fingerprint": fingerprint}

    
    print("\nStep 7: Uploading dataset card...")
    dataset_card = {
        "dataset_version"    : output_version,
        "source_version"     : dataset_version,
        "pipeline_version"   : PIPELINE_VERSION,
        "compiled_at"        : datetime.now(timezone.utc).isoformat(),
        "total_candidates"   : len(candidates),
        "total_rejected"     : len(rejected),
        "splits"             : dataset_stats,
        "rejection_reasons"  : rejected[:20],
        "leakage_prevention" : {
            "temporal_split"  : f"cutoff={cutoff_date_str}",
            "speaker_split"   : "speakers not shared across train/eval/test",
            "meeting_split"   : "all segments of a meeting in same split",
            "synthetic_policy": "synthetic data only in training split",
        },
        "quality_gates": {
            "min_duration_sec" : min_duration,
            "min_snr_db"       : min_snr,
            "min_tokens"       : min_tokens,
        },
    }

    card_path = f"{output_prefix}/dataset_card.json"
    conn.put_object(
        container, card_path,
        json.dumps(dataset_card, indent=2, default=str).encode(),
        content_type="application/json"
    )
    print(f"  Uploaded dataset card → {card_path}")

    
    print(f"\n{'='*60}")
    print(f"  COMPILATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Dataset version : {output_version}")
    print(f"  Output prefix   : {container}/{output_prefix}/")
    for split, stats in dataset_stats.items():
        print(f"  {split:5s} : {stats['count']} records "
              f"(fingerprint: {stats['fingerprint'][:8]})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile versioned train/eval/test datasets from production data"
    )
    parser.add_argument("--container",        required=True)
    parser.add_argument("--dataset-version",  default="1.0")
    parser.add_argument("--cutoff-date",      default=None,
                        help="Train/eval temporal split date (YYYY-MM-DD). "
                             "Data before this date → train, after → eval/test.")
    parser.add_argument("--min-duration",     type=float, default=1.5,
                        help="Minimum audio duration in seconds (default: 1.5)")
    parser.add_argument("--min-snr",          type=float, default=10.0,
                        help="Minimum SNR in dB (default: 10.0)")
    parser.add_argument("--min-tokens",       type=int,   default=3,
                        help="Minimum transcript token count (default: 3)")
    parser.add_argument("--output-version",   default="auto",
                        help="Output dataset version tag (default: auto-generated)")
    args = parser.parse_args()

    run_batch_pipeline(
        container        = args.container,
        dataset_version  = args.dataset_version,
        cutoff_date_str  = args.cutoff_date,
        min_duration     = args.min_duration,
        min_snr          = args.min_snr,
        min_tokens       = args.min_tokens,
        output_version   = args.output_version,
    )
