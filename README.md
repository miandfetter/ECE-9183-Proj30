# 🎥 MiroTalk + FastAPI ML Summarization System

---

## 📌 Overview

This project integrates:

* **MiroTalk (Node.js)** → real-time meeting + transcript (**Port 3000**)
* **FastAPI (Python)** → ML summarization backend (**Port 8000**)
* **Transformer-based model (PEFT + Hugging Face)**
* **Prometheus + Grafana** → monitoring

The system converts **meeting transcripts into summaries in real time**.

---

## 🏗️ System Architecture

```text
User (Browser)
     ↓
MiroTalk (Port 3000)
     ↓
Transcript
     ↓
FastAPI (/summarize) (Port 8000)
     ↓
ML Model
     ↓
Summary → UI
```

---

## 📂 Project Structure

```text
mirotalk/
├── main.py          # FastAPI backend
├── train/           # training logic
├── rollback/        # rollback logic
├── client/          # frontend
├── server/          # Node backend
├── widgets/         # UI components
├── public/          # static assets
├── src/             # core logic
```

---

# 🚀 How to Run the Project

## 🖥️ Step 1 — Connect to Chameleon VM (with Port Forwarding)

In a typical setup, we would connect using:

```bash
ssh -i ~/.ssh/r_mac cc@<VM-IP>
```

OR 

To ensure stable and reliable access, we used **SSH port forwarding**:

```bash
ssh -i ~/.ssh/r_mac \
  -L 3000:localhost:3000 \
  -L 8000:localhost:8000 \
  -L 9090:localhost:9090 \
  -L 3001:localhost:3001 \
  cc@129.114.25.170
```

---

## 🌐 Access Services (via Chameleon Cloud)

All services are running on the **Chameleon Cloud VM**, but are accessed locally using **SSH port forwarding**.

| Service    | Local URL (via SSH Tunnel) | Runs on Chameleon             |
| ---------- | -------------------------- | ----------------------------- |
| MiroTalk   | http://localhost/(VM_IP):000      | Port 3000                     |
| FastAPI    | http://localhost/(VM_IP):8000      | Port 8000                     |
| Prometheus | http://localhost/(VM_IP):9090      | Port 9090                     |
| Grafana    | http://localhost/(VM_IP):3001      | Port 3001 (Grafana container) |
| MLFlow    | http://localhost/(VM_IP):8002      | Port 8002  |      
---


---

### 🔁 Mapping Example

```text
Local Machine (Mac)         Chameleon VM
---------------------      ---------------------
localhost:3000   ───────→  MiroTalk (3000)
localhost:8000   ───────→  FastAPI (8000)
localhost:9090   ───────→  Prometheus (9090)
localhost:3001   ───────→  Grafana (3000)
```

## WebRTC Error Fix

MiroTalk uses WebRTC for real-time communication (audio/video). Modern browsers require WebRTC applications to run in a **secure context**, otherwise errors such as:

```text
The browser seems not supported by WebRTC

will occur.
During deployment on Chameleon Cloud, accessing the application via:
http://<VM-IP>:3000

caused WebRTC failures because public HTTP is not considered secure.

Solution Used
We implemented two working solutions to resolve the issue.

Option 1: SSH Port Forwarding (Localhost Access)
We used SSH tunneling to access the application via localhost:
ssh -i ~/.ssh/r_mac -L 3000:localhost:3000 cc@<VM-IP>

Then opened:
http://localhost:3000

Browsers treat localhost as a secure origin, so WebRTC works without requiring HTTPS.

Option 2: HTTPS using Nginx Reverse Proxy
We also configured HTTPS using a reverse proxy with Nginx.
Architecture:
Browser (HTTPS)
      ↓
Nginx (SSL termination)
      ↓
MiroTalk (localhost:3000)

Access:
https://<VM-IP>

We generated a self-signed SSL certificate for testing. This enables WebRTC but may show a browser warning.

Final Approach Used
We used both approaches:
SSH port forwarding (localhost) for development and debugging
HTTPS via Nginx for browser-based access without SSH

Result
After applying these fixes:
WebRTC browser error was resolved
Microphone and camera permissions worked
MiroTalk loaded successfully in all major browsers
Video conferencing worked as expected

Note
For production deployment, a domain name with a trusted SSL certificate (e.g., Let's Encrypt) should be used to remove browser warnings completely.

## 📦 Step 2 — Start MiroTalk (Port 3000)

```bash
cd ~/mirotalk
npm install
npm start
```

---

## 🧠 Step 3 — Start FastAPI Backend (Port 8000)

```bash
cd ~/mirotalk
python3 -m venv .venv
source .venv/bin/activate

python -m pip install fastapi uvicorn torch transformers peft sentencepiece accelerate

python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🔗 API Integration

```js
fetch("http://127.0.0.1:8000/summarize", {
```

---

# 🧠 Fallback Mechanism

To ensure reliability, a fallback mechanism is implemented.

```python
try:
    summary = model.generate(...)
except:
    summary = simple_summary(transcript)
```

### Why fallback is important

* ensures a response is always returned
* prevents API failures
* improves user experience

---

# 📊 Monitoring (Prometheus + Grafana)

## Architecture

```
FastAPI → /metrics → Prometheus → Grafana
```

---

## Metrics (FastAPI)

```python
from prometheus_client import Counter, Histogram

api_requests_total = Counter("api_requests_total", "Total API Requests")
request_latency = Histogram("request_latency_seconds", "Latency")
```

---

## Run Prometheus

```bash
docker run -d -p 9090:9090 \
-v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
prom/prometheus
```

---

## Run Grafana

```bash
docker run -d -p 3001:3000 grafana/grafana
```

---

## Grafana Queries

Requests/sec:

```
rate(api_requests_total[1m])
```

P95 latency:

```
histogram_quantile(0.95, sum(rate(request_latency_seconds_bucket[1m])) by (le))
```

---

# 📌 What We Monitor

| Metric            | Purpose     |
| ----------------- | ----------- |
| Request count     | system load |
| latency (p50/p95) | performance |
| throughput        | scalability |
| failures          | reliability |

---

# 🎯 Promotion & Rollback Logic

## Promote model if:

* latency remains stable
* error rate is low

## Rollback model if:

* error rate increases significantly
* latency exceeds threshold
* fallback usage increases

---

# 🎤 Demo Flow

1. Open MiroTalk
2. Start meeting
3. Speak → transcript generated
4. Click summary
5. FastAPI returns summary

---

# ⚠️ Important Notes

* Port **3000 → MiroTalk**
* Port **8000 → FastAPI**
* SSH tunneling is used due to network and security constraints

---

# 💬 Explanation (for demo)

> MiroTalk captures real-time transcripts and sends them to a FastAPI backend, where a transformer-based model generates summaries. A fallback mechanism ensures reliability, and Prometheus with Grafana is used to monitor system performance.

---

# ✅ Status

✔ MiroTalk running
✔ FastAPI working
✔ ML model integrated
✔ Fallback implemented
✔ Monitoring enabled
✔ End-to-end pipeline complete

---


# Proj30 — MiroTalk Meeting Summarization: Training Pipeline

## Overview

This repository contains the training pipeline for a BART-LoRA meeting summarization model built for MiroTalk. The model generates summaries of meeting transcripts.

The pipeline:
1. Pulls meeting transcripts from a Chameleon object store bucket (50 meetings)
2. Warm-starts from a MeetingBank pretrained LoRA adapter
3. Fine-tunes `facebook/bart-large-cnn` using LoRA on the MiroTalk meetings
4. Evaluates using ROUGE and gates registration against the current best model
5. Automatically promotes to production if it beats the current production model
6. Retrains automatically every Sunday at 2am via cron

---



## Prerequisites

- Access to [Chameleon Cloud](https://chameleoncloud.org) with project `CHI-251409`
- Two active leases on `KVM@TACC` (create at https://chi.tacc.chameleoncloud.org → Reservations → Leases):
  - **MLflow VM lease** — any `m1.medium` or larger flavor
  - **GPU training lease** — any GPU flavor (`g1.v100`, `g1.h100`, `g1.rtx6000`, etc.)
- A Trovi Jupyter instance to run the setup notebook

---

## Setup (One Time)

Open `infra/MLflow_GPU_setup.ipynb` on Trovi and run all cells top to bottom.

Before running, edit **Cell 1** with your lease names:
```python
MLFLOW_LEASE_NAME = "your-mlflow-lease-name"
GPU_LEASE_NAME    = "your-gpu-lease-name"
```

The notebook will automatically:
1. Generate EC2-compatible credentials for the Chameleon object store
2. Create the MLflow artifact bucket on `CHI@TACC`
3. Launch the MLflow VM on `KVM@TACC` and start MLflow + PostgreSQL via Docker Compose
4. Launch the GPU node on `KVM@TACC` (works with any GPU flavor)
5. Install Docker + NVIDIA toolkit on the GPU node
6. Clone this repo and build the training Docker image
7. Write the credentials file on the MLflow VM
8. Upload the SSH key so the cron trigger can SSH into the GPU node
9. Set up the weekly cron job
10. Run a smoke test to verify the full pipeline end to end


---

## About `/home/cc/.mlflow_s3_credentials`

This file lives on the MLflow VM and stores all environment variables needed
by the automated retraining pipeline. The cron job sources it before running
`retrain_trigger.py`. It is written automatically by the setup notebook and
is never committed to git.

| Variable | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | EC2-compatible key for Chameleon object store (auto-generated) |
| `AWS_SECRET_ACCESS_KEY` | EC2-compatible secret (auto-generated) |
| `MLFLOW_TRACKING_URI` | URL of the MLflow server |
| `MLFLOW_S3_ENDPOINT_URL` | Chameleon object store S3 endpoint (fixed) |
| `MLFLOW_ARTIFACTS_BUCKET` | Bucket where MLflow stores model artifacts |
| `TRAINING_DATA_BUCKET` | Bucket where meeting transcripts live |
| `GITHUB_REPO` | Repo cloned on the GPU node during automated retraining |
| `OS_PROJECT_NAME` | Chameleon project name |
| `GPU_NODE_IP` | Floating IP of the GPU node (SSH target for cron trigger) |
| `INFERENCE_SERVER_URL` | URL of the inference server `/reload` endpoint |

---

## Automated Retraining

A cron job fires every Sunday at 2am on the MLflow VM:

```
0 2 * * 0  set -a; source /home/cc/.mlflow_s3_credentials; set +a;
           python3 /home/cc/proj30/training/retrain_trigger.py >> /home/cc/retrain.log 2>&1
```

`retrain_trigger.py` SSHes from the MLflow VM into the GPU node and runs the training container. The GPU node must be running with an active lease for the cron job to succeed — if it is unreachable, the job logs the failure and exits gracefully, retrying the following Sunday.

To check retraining logs:
```bash
ssh cc@<mlflow-vm-ip> 'tail -f /home/cc/retrain.log'
```

To trigger retraining manually without waiting for Sunday, run Cell 18 in the setup notebook, or from any Trovi notebook:
```python
result = s.execute("""
    set -a
    source /home/cc/.mlflow_s3_credentials
    set +a
    python3 /home/cc/proj30/training/retrain_trigger.py
""")
print(result)
```

---

## Model Registry and Promotion

Models are tracked in MLflow at `http://<mlflow-vm-ip>:8000` under the name `bart-meeting-summarizer`.



**Manual promotion** (force a specific version to production):
```bash
MLFLOW_TRACKING_URI=http://<mlflow-ip>:8000 python3 training/promote.py
# or for a specific version:
MLFLOW_TRACKING_URI=http://<mlflow-ip>:8000 python3 training/promote.py --version 3
```

---

## Rollback

To roll back production to the previous version:
```bash
MLFLOW_TRACKING_URI=http://<mlflow-ip>:8000 python3 training/rollback.py
```

To roll back to a specific version:
```bash
MLFLOW_TRACKING_URI=http://<mlflow-ip>:8000 python3 training/rollback.py --version 2
```

---

## Gate

The gate prevents a degraded model from being registered. Each candidate model is compared against the current best staging model (not a fixed baseline), so production quality can only improve over time.

The committed baseline in `baseline_scores.json` is only used on the very first run when no staging model exists yet:

```json
{
  "rougeL": 0.18,
  "rouge1": 0.21,
  "rouge2": 0.09,
  "model": "facebook/bart-large-cnn",
  "notes": "Pretrained baseline, no fine-tuning. Used only on first run."
}
```

Gate logic: `candidate rougeL ≥ current_staging_rougeL − 0.02`

---

## Training Data

Meeting transcripts are stored in the Chameleon object store bucket `ObjStore_projecttranscriptionmirotalk` at `CHI@TACC` under the following structure:

```
raw/
  meeting_0000/
    transcript.json    {"transcript": "..."}
    metadata.json      {"summary": "...", "uid": "..."}
  meeting_0001/
  ...
```

The training script loads all meetings under `raw/` automatically. Currently 50 meetings are available.

---

## Known Limitations

- **Train = eval set**: With 50 meetings and no held-out test set, ROUGE scores reflect in-sample performance. This is an acknowledged limitation — a held-out test set would be required for a fair generalization estimate.
- **Manual GPU provisioning**: The GPU node must be manually provisioned via the setup notebook before the cron job can run. Due to a `chi` library compatibility issue on the MLflow VM, the trigger script cannot provision the GPU node automatically.
- **Chameleon lease expiry**: GPU leases expire. If a lease expires before Sunday, the cron job will fail gracefully until a new lease is created and the GPU node is reprovisioned using the setup notebook.
>>>>>>> d59d7a0df2427d7bd8b098efeb23fd0cc5dd56fd
