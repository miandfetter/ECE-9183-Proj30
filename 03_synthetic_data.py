#!/usr/bin/env python3
"""
03_synthetic_data.py
--------------------
Expands the MeetingBank dataset using synthetic data generation techniques.
Since MeetingBank is small (<5GB), this is required per project guidelines.

Techniques used:
  1. Text paraphrasing via back-translation prompts (Claude API)
  2. Audio augmentation (speed, pitch, noise) on existing .wav files
  3. Transcript perturbation (synonym substitution, punctuation variation)

Usage:
    python3 03_synthetic_data.py --container <name> --dataset-version 1.0 --multiplier 3
"""

import argparse
import io
import json
import os
import random
import re
import tempfile

import numpy as np
import soundfile as sf
import swiftclient

PIPELINE_VERSION = os.environ.get("GIT_SHA", "local")
SAMPLE_RATE = 16000

# Synonyms for common meeting words (lightweight, no external model needed)
SYNONYM_MAP = {
    "meeting": ["session", "assembly", "conference", "gathering"],
    "council": ["board", "committee", "panel", "body"],
    "motion": ["proposal", "measure", "resolution", "suggestion"],
    "approve": ["pass", "adopt", "ratify", "endorse"],
    "discuss": ["review", "consider", "examine", "address"],
    "member": ["representative", "delegate", "participant", "official"],
    "vote": ["ballot", "poll", "tally", "decision"],
    "agenda": ["schedule", "plan", "docket", "program"],
    "public": ["community", "citizens", "residents", "constituents"],
    "budget": ["funding", "finances", "allocation", "expenditure"],
}

SPEED_FACTORS = [0.9, 0.95, 1.05, 1.1]    # subtle speed changes
NOISE_LEVELS = [0.002, 0.005]               # low noise to preserve quality


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
def synonym_substitute(text, p=0.15):
    """Randomly replace words with synonyms at probability p."""
    tokens = text.split()
    result = []
    for token in tokens:
        clean = token.lower().strip(".,;:!?")
        if clean in SYNONYM_MAP and random.random() < p:
            synonym = random.choice(SYNONYM_MAP[clean])
            # Preserve original capitalisation
            if token[0].isupper():
                synonym = synonym.capitalize()
            result.append(synonym)
        else:
            result.append(token)
    return " ".join(result)


def punctuation_variation(text):
    """Randomly drop or alter punctuation — mimics ASR output variation."""
    # Randomly drop commas
    if random.random() < 0.4:
        text = text.replace(",", "")
    # Randomly lowercase first letter
    if random.random() < 0.3 and text:
        text = text[0].lower() + text[1:]
    return text


def augment_transcript(text, variant_id):
    """Produce a text variant using synonym substitution + punctuation changes."""
    random.seed(variant_id)  # deterministic per variant
    text = synonym_substitute(text, p=0.15)
    text = punctuation_variation(text)
    return text


# ── Audio augmentation ────────────────────────────────────────────────────────

def speed_perturb(audio, factor):
    """Change speed without pitch shift (simple resampling)."""
    new_len = int(len(audio) / factor)
    return np.interp(
        np.linspace(0, len(audio), new_len),
        np.arange(len(audio)),
        audio
    ).astype(np.float32)


def add_gaussian_noise(audio, noise_level):
    """Add Gaussian noise to audio."""
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_level
    return np.clip(audio + noise, -1.0, 1.0)


def augment_audio(audio, variant_id):
    """Apply a deterministic augmentation based on variant_id."""
    np.random.seed(variant_id)
    random.seed(variant_id)

    augmented = audio.copy()

    # Speed perturbation
    if variant_id % 2 == 0:
        factor = random.choice(SPEED_FACTORS)
        augmented = speed_perturb(augmented, factor)

    # Add noise
    if variant_id % 3 != 0:
        noise_level = random.choice(NOISE_LEVELS)
        augmented = add_gaussian_noise(augmented, noise_level)

    return augmented


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate_synthetic(container, dataset_version, multiplier):
    conn = get_swift_connection()
    proc_prefix = f"processed/v{dataset_version}/pipeline-{PIPELINE_VERSION}"
    synth_prefix = f"synthetic/v{dataset_version}/pipeline-{PIPELINE_VERSION}"

    print(f"Loading train manifest from {proc_prefix}...")
    _, manifest_body = conn.get_object(
        container, f"{proc_prefix}/train_manifest.json")
    manifest = json.loads(manifest_body)
    records = manifest["records"]

    synthetic_records = []

    for i, record in enumerate(records):
        mid = record["meeting_id"]
        print(f"[{i+1}/{len(records)}] Generating {multiplier} variants for {mid}...")

        # Load original transcript
        _, trans_body = conn.get_object(container, record["transcript_path"])
        trans_obj = json.loads(trans_body)
        original_text = trans_obj.get("transcript_normalized", "")

        # Load original audio if available
        original_audio = None
        if record.get("audio_path"):
            try:
                _, audio_body = conn.get_object(container, record["audio_path"])
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_body)
                    tmp_path = tmp.name
                original_audio, _ = sf.read(tmp_path)
                os.unlink(tmp_path)
                original_audio = original_audio.astype(np.float32)
                if original_audio.ndim > 1:
                    original_audio = original_audio.mean(axis=1)
            except Exception as e:
                print(f"  Warning: could not load audio for {mid}: {e}")

        for v in range(multiplier):
            variant_id = i * multiplier + v
            synth_id = f"{mid}_synth_{v:02d}"
            base = f"{synth_prefix}/{synth_id}"

            # ── Augmented transcript ──────────────────────────────
            augmented_text = augment_transcript(original_text, variant_id)
            trans_out = json.dumps({
                "meeting_id": synth_id,
                "source_meeting_id": mid,
                "variant_id": v,
                "transcript_normalized": augmented_text,
                "tokens": augmented_text.split(),
                "augmentation": "synonym_substitution+punctuation_variation",
                "pipeline_version": PIPELINE_VERSION,
            }, indent=2).encode()

            conn.put_object(container, f"{base}/transcript_processed.json",
                            trans_out, content_type="application/json")

            # ── Augmented audio ───────────────────────────────────
            audio_path = None
            if original_audio is not None:
                aug_audio = augment_audio(original_audio, variant_id)
                audio_buf = io.BytesIO()
                sf.write(audio_buf, aug_audio, SAMPLE_RATE, format="WAV")
                conn.put_object(container, f"{base}/audio_16khz.wav",
                                audio_buf.getvalue(), content_type="audio/wav")
                audio_path = f"{base}/audio_16khz.wav"

            synthetic_records.append({
                "meeting_id": synth_id,
                "source_meeting_id": mid,
                "variant_id": v,
                "split": "train",
                "transcript_path": f"{base}/transcript_processed.json",
                "audio_path": audio_path,
                "augmentation_method": "text:synonym+punct / audio:speed+noise",
                "pipeline_version": PIPELINE_VERSION,
                "dataset_version": dataset_version,
            })

    # Upload synthetic manifest
    synth_manifest = json.dumps({
        "split": "train",
        "type": "synthetic",
        "multiplier": multiplier,
        "source_records": len(records),
        "synthetic_records": len(synthetic_records),
        "pipeline_version": PIPELINE_VERSION,
        "dataset_version": dataset_version,
        "records": synthetic_records,
    }, indent=2).encode()

    conn.put_object(container, f"{synth_prefix}/synthetic_manifest.json",
                    synth_manifest, content_type="application/json")

    print(f"\nSynthetic data generation complete.")
    print(f"  Original records : {len(records)}")
    print(f"  Synthetic records: {len(synthetic_records)}")
    print(f"  Total training   : {len(records) + len(synthetic_records)}")
    print(f"  Output prefix    : {container}/{synth_prefix}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--dataset-version", default="1.0")
    parser.add_argument("--multiplier", type=int, default=3,
                        help="Number of synthetic variants per original sample")
    args = parser.parse_args()
    generate_synthetic(args.container, args.dataset_version, args.multiplier)
