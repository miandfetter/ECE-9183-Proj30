# Data Pipeline — MeetingBank Transcription Service
**Project 30 | Data Role | ECE-9183**

End-to-end data pipeline for ingesting, preprocessing, validating, and monitoring MeetingBank data on Chameleon Cloud object storage.

---

## Quick Start

```bash
# 1. Set up credentials (run every session)
source ~/setup.sh

# 2. Run full pipeline
python3 run_pipeline.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --split test --limit 500 --multiplier 3

# 3. Run all validators
python3 01_ingestion_validator.py --container ObjStore_projecttranscriptionmirotalk --dataset-version 1.0 --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns
python3 02_training_set_validator.py --container ObjStore_projecttranscriptionmirotalk --dataset-version <version> --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns
python3 03_drift_monitor.py --container ObjStore_projecttranscriptionmirotalk --reference-version 1.0 --mode once --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns
python3 04_collect_feedback.py --container ObjStore_projecttranscriptionmirotalk --min-confidence 0.85 --days 7 --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns
```

---

## Setup

### 1. Create application credentials
1. Go to **chi.tacc.chameleoncloud.org → Settings → Application Credentials**
2. Create new credential → download openrc file
3. Create `~/setup.sh`:

```bash
cat > ~/setup.sh << 'EOF'
export OS_AUTH_URL="https://chi.tacc.chameleoncloud.org:5000/v3"
export OS_APPLICATION_CREDENTIAL_ID="your-credential-id"
export OS_APPLICATION_CREDENTIAL_SECRET="your-credential-secret"
export OS_REGION_NAME="CHI@TACC"
export MLFLOW_TRACKING_URI="file:///home/cc/ECE-9183-Proj30/mlruns"
echo "Environment ready!"
EOF

source ~/setup.sh
```

### 2. Install dependencies
```bash
pip install datasets soundfile python-swiftclient \
    python-keystoneclient keystoneauth1 mlflow \
    fastapi uvicorn python-multipart numpy requests \
    --break-system-packages --quiet
```

### 3. Verify connection
```bash
swift \
  --os-auth-type v3applicationcredential \
  --os-auth-url $OS_AUTH_URL \
  --os-application-credential-id $OS_APPLICATION_CREDENTIAL_ID \
  --os-application-credential-secret "$OS_APPLICATION_CREDENTIAL_SECRET" \
  list ObjStore_projecttranscriptionmirotalk
```

---

## Scripts

### `run_pipeline.py` — Full pipeline runner
Runs all three stages: ingest → preprocess → synthetic data generation.

```bash
# Full pipeline (recommended)
MLFLOW_TRACKING_URI=file:///home/cc/ECE-9183-Proj30/mlruns \
python3 run_pipeline.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --split test \
    --limit 500 \
    --multiplier 3

# Arguments
#   --container     Object storage container name
#   --split         HuggingFace split: train | validation | test
#   --limit         Number of meetings to ingest (default: 20)
#   --multiplier    Synthetic data multiplier (default: 3)
#   --stage         Start from stage N: 1=ingest, 2=preprocess, 3=synthetic
```

---

### `01_ingest.py` — Ingest from HuggingFace
Downloads MeetingBank and uploads to object storage.

```bash
python3 01_ingest.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --split test \
    --limit 500

# Output: raw/v1.0/{meeting_id}/transcript.json
#         raw/v1.0/{meeting_id}/metadata.json
#         raw/v1.0/manifest.json
```

---

### `02_preprocess.py` — Preprocess and split
Normalizes transcripts, creates 70/15/15 train/val/test splits.

```bash
python3 02_preprocess.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --dataset-version 1.0

# Output: processed/v1.0/pipeline-local/{split}/{meeting_id}/transcript_processed.json
#         processed/v1.0/pipeline-local/train_manifest.json
#         processed/v1.0/pipeline-local/val_manifest.json
#         processed/v1.0/pipeline-local/test_manifest.json
```

---

### `03_synthetic_data.py` — Generate synthetic data
Augments training data using synonym substitution and punctuation variation.

```bash
python3 03_synthetic_data.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --dataset-version 1.0 \
    --multiplier 3

# Output: synthetic/v1.0/pipeline-local/{meeting_id}_synth_{n}/
#         synthetic/v1.0/pipeline-local/synthetic_manifest.json
```

---

### `batch_pipeline.py` — Compile final datasets
Applies quality gates, temporal+speaker splits, and compiles versioned datasets for training.

```bash
python3 batch_pipeline.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --dataset-version 1.0 \
    --cutoff-date 2021-06-01 \
    --min-duration 0.0 \
    --min-snr 0.0 \
    --output-version auto

# Arguments
#   --cutoff-date    Temporal split date (train=before, eval=after)
#   --min-duration   Minimum audio duration in seconds (default: 1.5)
#   --min-snr        Minimum SNR in dB (default: 10.0)
#   --output-version Dataset version tag (auto = timestamp_gitsha)

# Output: datasets/v{version}/train_dataset.json  (140 records)
#         datasets/v{version}/eval_dataset.json   (7 records)
#         datasets/v{version}/test_dataset.json   (8 records)
#         datasets/v{version}/dataset_card.json
```

---

### `01_ingestion_validator.py` — Validate at ingestion
Runs quality checks on raw ingested data. Gates pipeline if >20% quarantined.

```bash
MLFLOW_TRACKING_URI=file:///home/cc/ECE-9183-Proj30/mlruns \
python3 01_ingestion_validator.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --dataset-version 1.0 \
    --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns

# Checks: schema completeness, transcript length/quality,
#         audio validity, checksum integrity
# Gate:   fails if quarantine rate > 20%
# Logs:   MLflow experiment: data_quality_ingestion
```

---

### `02_training_set_validator.py` — Validate before training
Validates compiled datasets before handing to training team. Hard fails on data leakage.

```bash
# First check what dataset version exists
swift \
  --os-auth-type v3applicationcredential \
  --os-auth-url $OS_AUTH_URL \
  --os-application-credential-id $OS_APPLICATION_CREDENTIAL_ID \
  --os-application-credential-secret "$OS_APPLICATION_CREDENTIAL_SECRET" \
  list ObjStore_projecttranscriptionmirotalk --prefix datasets/

# Then run with that version
MLFLOW_TRACKING_URI=file:///home/cc/ECE-9183-Proj30/mlruns \
python3 02_training_set_validator.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --dataset-version 20260419_160121_local \
    --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns

# Checks: min size, data leakage, synthetic ratio,
#         speaker balance, vocabulary coverage
# Gate:   hard fail on leakage or insufficient size
# Logs:   MLflow experiment: data_quality_training_set
```

---

### `03_drift_monitor.py` — Monitor production drift
Detects distribution shift between training data and live inference data.

```bash
# Run once
MLFLOW_TRACKING_URI=file:///home/cc/ECE-9183-Proj30/mlruns \
python3 03_drift_monitor.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --reference-version 1.0 \
    --mode once \
    --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns

# Run continuously every 5 minutes
MLFLOW_TRACKING_URI=file:///home/cc/ECE-9183-Proj30/mlruns \
python3 03_drift_monitor.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --reference-version 1.0 \
    --mode continuous \
    --interval 300 \
    --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns

# Arguments
#   --mode              once | continuous
#   --interval          Seconds between checks in continuous mode (default: 300)
#   --reference-version Dataset version to compare against

# Reads:  inference_logs/{date}/{request_id}.json
# Alerts: WARNING at 30% drift, CRITICAL at 60%
# Logs:   MLflow experiment: data_quality_drift_monitor
# Output: /tmp/drift_status.json (Docker health check)
```

---

### `04_collect_feedback.py` — Collect production feedback
Harvests high-confidence inference transcripts for retraining.

```bash
MLFLOW_TRACKING_URI=file:///home/cc/ECE-9183-Proj30/mlruns \
python3 04_collect_feedback.py \
    --container ObjStore_projecttranscriptionmirotalk \
    --min-confidence 0.85 \
    --days 7 \
    --mlflow-uri file:///home/cc/ECE-9183-Proj30/mlruns

# Arguments
#   --min-confidence    Minimum model confidence to include (default: 0.85)
#   --days              Number of past days to scan (default: 7)

# Reads:  inference_logs/{date}/{request_id}.json
# Output: feedback/collected_{timestamp}.json
# Logs:   MLflow experiment: data_quality_feedback
```

---

### `feature_service.py` — Online feature extraction
FastAPI service that converts raw audio to log-mel spectrograms for ASR inference.

```bash
# Start service
PORT=5001 PYTHONPATH=/home/cc/ECE-9183-Proj30 \
python3 feature_service.py &

# Health check
curl http://localhost:5001/health

# Extract features from audio
curl -X POST http://localhost:5001/extract \
    -F "audio=@your_audio.wav"

# View metrics
curl http://localhost:5001/metrics

# Generate test audio for testing
python3 -c "
import wave, struct, math, io
sr = 16000
samples = [int(32767 * math.sin(2*math.pi*220*i/sr)) for i in range(48000)]
buf = io.BytesIO()
with wave.open(buf, 'wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
    f.writeframes(struct.pack('<'+'h'*len(samples), *samples))
with open('test.wav', 'wb') as f:
    f.write(buf.getvalue())
print('test.wav created')
"

curl -X POST http://localhost:5001/extract -F "audio=@test.wav"

# Returns: {"shape": [80, 3000], "features_b64": "...", "duration_sec": 3.0}
# Serving team calls this before inference:
#   POST http://localhost:5001/extract with audio/wav file
```

---

## Object Storage Structure

```
ObjStore_projecttranscriptionmirotalk/
├── raw/v1.0/
│   ├── manifest.json              ← dataset manifest
│   └── meeting_0000/
│       ├── transcript.json        ← raw transcript
│       ├── metadata.json          ← meeting metadata
│       └── validation_result.json ← quality check result
├── processed/v1.0/pipeline-local/
│   ├── train_manifest.json
│   ├── val_manifest.json
│   ├── test_manifest.json
│   └── train/meeting_0000/
│       └── transcript_processed.json
├── synthetic/v1.0/pipeline-local/
│   ├── synthetic_manifest.json
│   └── meeting_0000_synth_00/
│       └── transcript_processed.json
├── datasets/v{version}/
│   ├── train_dataset.json         ← training team reads this
│   ├── eval_dataset.json
│   ├── test_dataset.json
│   └── dataset_card.json
├── inference_logs/{date}/
│   └── {request_id}.json          ← serving team writes this
├── feedback/
│   └── collected_{timestamp}.json ← collected for retraining
└── signals/
    └── retrain_needed.json        ← drift trigger signal
```

---

## MLflow Experiments

All quality metrics logged to MLflow at:
```
file:///home/cc/ECE-9183-Proj30/mlruns
```

| Experiment | Script | Key Metrics |
|---|---|---|
| `data_quality_ingestion` | 01_ingestion_validator.py | pass_rate, quarantine_rate, quality_gate_passed |
| `data_quality_training_set` | 02_training_set_validator.py | train_count, synthetic_ratio, leakage_detected |
| `data_quality_drift_monitor` | 03_drift_monitor.py | max_drift_score, n_alerts, drift_status |
| `data_quality_feedback` | 04_collect_feedback.py | collected, rejected, collection_rate |

---

## Integration with Team

### Training team reads datasets from:
```
datasets/v20260419_160121_local/train_dataset.json
→ 140 records (35 real + 105 synthetic)
```

### Serving team calls feature service at:
```
POST http://localhost:5001/extract
Content-Type: multipart/form-data
Body: audio=<wav file>
Returns: {"shape": [80, 3000], "features_b64": "..."}
```

### Serving team writes inference logs to:
```
inference_logs/{YYYY-MM-DD}/{request_id}.json
Format: {request_id, transcript, confidence, duration_sec, latency_ms, model_version, timestamp}
```

### Drift monitor writes retraining signal to:
```
signals/retrain_needed.json
→ Training team's retrain_trigger.py checks for this file
```

---

## Docker

```bash
# Build
docker build -t proj30-data .

# Run full pipeline
docker run --rm \
  -e OS_AUTH_URL=https://chi.tacc.chameleoncloud.org:5000/v3 \
  -e OS_APPLICATION_CREDENTIAL_ID=<your-id> \
  -e OS_APPLICATION_CREDENTIAL_SECRET=<your-secret> \
  -e OS_REGION_NAME=CHI@TACC \
  proj30-data \
  --container ObjStore_projecttranscriptionmirotalk \
  --limit 500 --multiplier 3

# Run feature service
docker run -d -p 5001:5001 \
  -e PORT=5001 \
  proj30-data \
  python3 feature_service.py

# Run drift monitor continuously
docker run -d --name drift-monitor \
  -e OS_AUTH_URL=... \
  -e OS_APPLICATION_CREDENTIAL_ID=... \
  -e OS_APPLICATION_CREDENTIAL_SECRET=... \
  proj30-data \
  python3 03_drift_monitor.py \
  --container ObjStore_projecttranscriptionmirotalk \
  --mode continuous --interval 300
```

---

## Common Issues

| Error | Fix |
|---|---|
| `Unauthorized. Check application credential` | Run `source ~/setup.sh` — token expired |
| `ModuleNotFoundError` | Run `pip install <package> --break-system-packages` |
| `No manifest found` | Run `01_ingest.py` first |
| `Quality gate FAILED: duplicate_meeting_id` | Re-ingestion is intentional — validator is already fixed |
| `Port already allocated` | Run `pkill -f feature_service` then restart |
