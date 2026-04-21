#!/usr/bin/env python3
"""
01_ingest.py
------------
Ingests MeetingBank dataset from HuggingFace into Chameleon Object Storage.
Saves raw audio, transcripts, and metadata with versioned prefixes.

Usage:
    python3 01_ingest.py --container <container_name> --split test --limit 50
"""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime

import soundfile as sf
import swiftclient
from datasets import load_dataset

DATASET_NAME = "huuuyeah/meetingbank"
DATASET_VERSION = "1.0"  # bump this when HF dataset updates


def get_swift_connection():
    """Connect to Chameleon Object Storage using application credentials."""
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


def upload_bytes(conn, container, object_name, data, content_type="application/octet-stream"):
    """Upload bytes to Swift object storage."""
    conn.put_object(container, object_name, data, content_type=content_type)
    print(f"  Uploaded: {object_name}")


def ingest(container, split, limit, dry_run=False):
    print(f"Loading MeetingBank ({split}, limit={limit}) from HuggingFace...")
    ds = load_dataset(DATASET_NAME, split=f"{split}[:{limit}]", trust_remote_code=True)

    conn = None if dry_run else get_swift_connection()

    # Ensure container exists and is public
    if not dry_run:
        try:
            conn.put_container(container, headers={"X-Container-Read": ".r:*"})
        except Exception:
            pass  # already exists

    manifest = []
    prefix = f"raw/v{DATASET_VERSION}"
    ingest_ts = datetime.utcnow().isoformat()

    for i, row in enumerate(ds):
        meeting_id = row.get("meeting_id", f"meeting_{i:04d}")
        base = f"{prefix}/{meeting_id}"

        # ── Transcript ────────────────────────────────────────────
        transcript_data = json.dumps({
            "meeting_id": meeting_id,
            "transcript": row.get("transcript", ""),
            "words": row.get("words", []),
        }, indent=2).encode()

        if not dry_run:
            upload_bytes(conn, container, f"{base}/transcript.json",
                         transcript_data, "application/json")

        # ── Metadata ──────────────────────────────────────────────
        meta = {k: v for k, v in row.items()
                if k not in ["audio", "transcript", "words"]}
        meta.update({
            "meeting_id": meeting_id,
            "ingest_timestamp": ingest_ts,
            "source_dataset": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "split": split,
        })
        meta_data = json.dumps(meta, indent=2, default=str).encode()

        if not dry_run:
            upload_bytes(conn, container, f"{base}/metadata.json",
                         meta_data, "application/json")

        # ── Audio ─────────────────────────────────────────────────
        audio_path = None
        if "audio" in row and row["audio"] is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, row["audio"]["array"],
                         row["audio"]["sampling_rate"])
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_path)

            audio_path = f"{base}/audio.wav"
            if not dry_run:
                upload_bytes(conn, container, audio_path,
                             audio_bytes, "audio/wav")

        # ── Manifest entry ────────────────────────────────────────
        manifest.append({
            "meeting_id": meeting_id,
            "transcript_path": f"{base}/transcript.json",
            "metadata_path": f"{base}/metadata.json",
            "audio_path": audio_path,
            "checksum": hashlib.md5(transcript_data).hexdigest(),
        })
        print(f"[{i+1}/{limit}] Processed {meeting_id}")

    # Upload manifest
    manifest_data = json.dumps({
        "dataset_version": DATASET_VERSION,
        "split": split,
        "ingest_timestamp": ingest_ts,
        "record_count": len(manifest),
        "records": manifest,
    }, indent=2).encode()

    if not dry_run:
        upload_bytes(conn, container, f"{prefix}/manifest.json",
                     manifest_data, "application/json")

    print(f"\nIngestion complete. {len(manifest)} meetings uploaded to {container}/{prefix}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ingest(args.container, args.split, args.limit, args.dry_run)
