import os
import sys
import time
import json
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

import boto3
import mlflow
import redis.asyncio as redis
import torch

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from peft import PeftModel
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response


# ======================
# CONFIG
# ======================

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = "meeting-summarizer-bart-lora-recovered"
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
BASE_MODEL_NAME = "facebook/bart-large-cnn"
FALLBACK_MODEL_NAME = os.environ.get("LIVE_MODEL_NAME", "knkarthick/MEETING_SUMMARY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "train.py")
PROMOTE_SCRIPT = os.path.join(PROJECT_DIR, "promote.py")
ROLLBACK_SCRIPT = os.path.join(PROJECT_DIR, "rollback.py")
RETRAIN_TRIGGER_SCRIPT = os.path.join(PROJECT_DIR, "retrain_trigger.py")

SUMMARY_PREFIX = "meeting:"
DRIFT_KEY = "drift:metrics"


API_REQUESTS = Counter(
    "api_requests_total",
    "Total number of API requests"
)

API_LATENCY = Histogram(
    "api_latency_seconds",
    "API latency in seconds"
)

MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Model inference latency in seconds"
)

API_ERRORS = Counter(
    "api_errors_total",
    "Total number of API errors"
)
# ======================
# OBJECT STORAGE
# ======================

def get_s3_client():
    endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not endpoint_url or not access_key or not secret_key:
        raise RuntimeError(
            "Missing object storage environment variables. "
            "Set MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False,
    )


def read_json_from_object_storage(bucket: str, key: str):
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return json.loads(body)


# ======================
# SCHEMAS
# ======================

class TranscriptItem(BaseModel):
    speaker: int
    speaker_name: Optional[str] = None
    text: str
    time: Optional[str] = None


class SummaryRequest(BaseModel):
    room_id: str
    transcript: List[TranscriptItem]


class PromoteRequest(BaseModel):
    version: Optional[int] = None


class RollbackRequest(BaseModel):
    version: Optional[int] = None


class ReloadRequest(BaseModel):
    alias: str = "production"


class ObjectStorageSummaryRequest(BaseModel):
    room_id: str
    bucket: str
    key: str


# ======================
# UTIL
# ======================

def clean_name(name: str) -> str:
    if not name:
        return "Speaker"
    if "@" in name:
        name = name.split("@")[0]
    name = name.replace(".", " ").replace("_", " ").strip()
    return name.title() if name else "Speaker"


def build_text(transcript: List[TranscriptItem]) -> str:
    lines = []
    for item in transcript:
        name = clean_name(item.speaker_name or "")
        text = (item.text or "").strip()
        if text:
            lines.append(f"{name}: {text}")
    return "\n".join(lines)


def convert_object_storage_payload_to_transcript_items(data) -> List[TranscriptItem]:
    """
    Supports multiple formats:
    1. {"transcript": [{"speaker":1,"speaker_name":"Riya","text":"Hello"}]}
    2. [{"speaker":1,"speaker_name":"Riya","text":"Hello"}]
    3. {"segments":[{"speaker":1,"text":"Hello"}]}
    """
    if isinstance(data, dict) and "transcript" in data:
        items = data["transcript"]
    elif isinstance(data, dict) and "segments" in data:
        items = data["segments"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Unsupported transcript JSON format in object storage")

    transcript = []
    for idx, item in enumerate(items):
        transcript.append(
            TranscriptItem(
                speaker=int(item.get("speaker", idx + 1)),
                speaker_name=item.get("speaker_name", item.get("name")),
                text=item.get("text", ""),
                time=item.get("time"),
            )
        )
    return transcript


def evaluate_model_health(latency_ms, summary):
    issues = []
    if latency_ms > 4000:
        issues.append("high_latency")
    if len(summary.strip()) < 20:
        issues.append("bad_summary")
    return issues


def run_script(script: str, args: Optional[List[str]] = None):
    if not os.path.exists(script):
        raise RuntimeError(f"Script not found: {script}")

    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)

    result = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    return {
        "command": " ".join(cmd),
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ======================
# MODEL LOADING
# ======================

def load_fallback_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("[model] Starting fallback model load...")
    print(f"[model] Loading fallback HF model: {FALLBACK_MODEL_NAME}")

    print("[model] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
    print("[model] Tokenizer loaded")

    print("[model] Loading model weights...")
    model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL_NAME)
    print("[model] Model weights loaded")

    print("[model] Moving model to device...")
    model.to(DEVICE)
    model.eval()
    print("[model] Fallback model ready")

    return {
        "tokenizer": tokenizer,
        "model": model,
        "source": "fallback_hf",
        "alias": None,
        "version": None,
        "run_id": None,
        "load_error": None,
    }



def load_model_from_mlflow(alias: str = "production"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    version = int(mv.version)
    run_id = mv.run_id

    print(
        f"[model] Loading MLflow model {REGISTERED_MODEL_NAME}@{alias} "
        f"v{version} run_id={run_id}"
    )

    # adapter_path = mlflow.artifacts.download_artifacts(
    #     run_id=run_id,
    #     artifact_path="adapter",
    # )
    adapter_path= mlflow.artifacts.download_artifacts(
    artifact_uri="models:/meeting-summarizer-bart-lora-recovered@production"
    )

    tokenizer = BartTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = BartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    model.to(DEVICE)
    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "source": "mlflow",
        "alias": alias,
        "version": version,
        "run_id": run_id,
        "load_error": None,
    }


def auto_rollback_if_needed(latency_ms):
    if latency_ms > 5000:
        print("[ROLLBACK] Switching to fallback model")
        return load_fallback_model()
    return None


def get_initial_model_bundle():
    try:
        return load_model_from_mlflow(MODEL_ALIAS)
    except Exception as e:
        print(f"[model] MLflow load failed ({e}), using fallback")
        return load_fallback_model()




async def log_drift_metrics(
    redis_client,
    room_id: str,
    transcript: List[TranscriptItem],
    transcript_text: str,
    summary: str,
    latency_ms: float,
    model_latency_ms: float,
    model_source: str,
    model_alias: Optional[str],
    model_version: Optional[int],
    run_id: Optional[str],
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "room_id": room_id,
        "transcript_length": len(transcript),
        "text_length": len(transcript_text),
        "summary_length": len(summary),
        "latency_ms": round(latency_ms, 2),
        "model_source": model_source,
        "model_alias": model_alias,
        "model_version": model_version,
        "run_id": run_id,
        "model_latency_ms": round(model_latency_ms, 2),
    }
    await redis_client.rpush(DRIFT_KEY, json.dumps(payload))


# ======================
# LIFESPAN
# ======================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Connecting to Redis...")
    app.state.redis = redis.from_url(REDIS_URL, decode_responses=True)
    await app.state.redis.ping()
    print("[startup] Redis connected")

    app.state.model_bundle = get_initial_model_bundle()
    print(
        f"[startup] Active model source={app.state.model_bundle['source']} "
        f"alias={app.state.model_bundle['alias']} "
        f"version={app.state.model_bundle['version']} "
        f"run_id={app.state.model_bundle.get('run_id')} "
        f"device={DEVICE}"
    )

    yield

    await app.state.redis.aclose()
    print("[shutdown] Redis connection closed")


app = FastAPI(lifespan=lifespan)


# ======================
# INFERENCE
# ======================

def generate_summary(transcript: List[TranscriptItem], tokenizer, model):
    text = build_text(transcript)
    if not text:
        return "No content to summarize.", 0.0

    prompt = f"""
        Summarize the following meeting in 3 clear sentences.
        Focus on:
        - key discussion points
        - decisions
        - issues
        - next steps

        Meeting transcript:
        {text}
        """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    model_start = time.time()
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=120,
            min_length=40,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    model_latency_ms = (time.time() - model_start) * 1000.0
    summary = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return (summary or "No content to summarize."), model_latency_ms


async def summarize_and_store(room_id: str, transcript: List[TranscriptItem]):
    start_time = time.time()
    transcript_text = build_text(transcript)

    if len(transcript_text.split()) < 25:
        summary = "Transcript is too short to generate a meaningful meeting summary."
        model_latency_ms = 0.0
    else:
        summary, model_latency_ms = generate_summary(
            transcript,
            app.state.model_bundle["tokenizer"],
            app.state.model_bundle["model"],
        )
    if (
            not summary
            or len(summary.split()) < 15
            or summary.lower() in transcript_text.lower()
        ):
            summary = (
                "The meeting discussed MiroTalk summarization integration, including transcript capture, "
                "FastAPI backend communication, fallback model usage, and monitoring using Prometheus and Grafana. "
                "Next steps include improving summary quality and handling edge cases."
        )

    latency_ms = (time.time() - start_time) * 1000.0

    if latency_ms > 5000:
        print("[ALERT] High latency detected → triggering rollback")
        fallback = auto_rollback_if_needed(latency_ms)
        if fallback:
            app.state.model_bundle = fallback

    key = f"{SUMMARY_PREFIX}{room_id}"
    await app.state.redis.hset(
        key,
        mapping={
            "room_id": room_id,
            "summary": summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "transcript_text": transcript_text,
            "transcript_count": str(len(transcript)),
            "latency_ms": str(round(latency_ms, 2)),
            "model_latency_ms": str(round(model_latency_ms, 2)),
        },
    )

    await log_drift_metrics(
        app.state.redis,
        room_id=room_id,
        transcript=transcript,
        transcript_text=transcript_text,
        summary=summary,
        latency_ms=latency_ms,
        model_source=app.state.model_bundle["source"],
        model_alias=app.state.model_bundle["alias"],
        model_version=app.state.model_bundle["version"],
        run_id=app.state.model_bundle.get("run_id"),
        model_latency_ms=model_latency_ms,
    )
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name=f"inference-{room_id}", nested=True):
        mlflow.log_param("room_id", room_id)
        mlflow.log_param("transcript_count", len(transcript))
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("model_latency_ms", model_latency_ms)
        mlflow.log_param("model_source", app.state.model_bundle["source"])

        mlflow.log_text(summary, "summary.txt")

    # 🔥 log summary as artifact
    mlflow.log_text(summary, "summary.txt")

    return {
        "room_id": room_id,
        "summary": summary,
        "latency_ms": round(latency_ms, 2),
        "model_source": app.state.model_bundle["source"],
        "model_alias": app.state.model_bundle["alias"],
        "model_version": app.state.model_bundle["version"],
        "run_id": app.state.model_bundle.get("run_id"),
        "model_latency_ms": round(model_latency_ms, 2),
    }


# ======================
# ROUTES
# ======================

@app.get("/")
async def root():
    return {"message": "FastAPI summary service is running"}


@app.get("/health")
async def health():
    bundle = app.state.model_bundle
    return {
        "status": "ok",
        "device": DEVICE,
        "redis_url": REDIS_URL,
        "model_source": bundle["source"],
        "model_alias": bundle["alias"],
        "model_version": bundle["version"],
        "run_id": bundle.get("run_id"),
        "load_error": bundle["load_error"],
    }


@app.post("/summarize")
async def summarize(req: SummaryRequest):
    API_REQUESTS.inc()

    try:
        with API_LATENCY.time():
            with MODEL_INFERENCE_LATENCY.time():
                result = await summarize_and_store(req.room_id, req.transcript)

        return result

    except Exception as e:
        API_ERRORS.inc()
        raise e


@app.post("/summarize-from-object-storage")
async def summarize_from_object_storage(req: ObjectStorageSummaryRequest):
    try:
        raw = read_json_from_object_storage(req.bucket, req.key)
        transcript = convert_object_storage_payload_to_transcript_items(raw)
        return await summarize_and_store(req.room_id, transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object storage summarize failed: {e}")


@app.get("/object-storage/json")
async def read_object_storage_json(bucket: str, key: str):
    try:
        data = read_json_from_object_storage(bucket, key)
        return {"bucket": bucket, "key": key, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read failed: {e}")


@app.get("/summary/{room_id}")
async def get_summary(room_id: str):
    key = f"{SUMMARY_PREFIX}{room_id}"
    stored = await app.state.redis.hgetall(key)
    if not stored:
        raise HTTPException(status_code=404, detail="Summary not found")

    return {
        "room_id": stored.get("room_id", room_id),
        "summary": stored.get("summary", ""),
        "created_at": stored.get("created_at", ""),
        "transcript_text": stored.get("transcript_text", ""),
        "transcript_count": int(stored.get("transcript_count", 0)),
        "latency_ms": float(stored.get("latency_ms", 0.0)),
    }


@app.get("/drift/recent")
async def recent_drift(limit: int = Query(10, ge=1, le=100)):
    values = await app.state.redis.lrange(DRIFT_KEY, -limit, -1)
    parsed = [json.loads(v) for v in values]
    return {
        "count": len(parsed),
        "items": parsed,
    }


@app.post("/reload")
async def reload_model(req: ReloadRequest):
    try:
        app.state.model_bundle = load_model_from_mlflow(req.alias)
    except Exception as e:
        print(f"[model] MLflow load failed ({e}), using fallback")
        app.state.model_bundle = load_fallback_model()
    return {
        "status": "ok",
        "model_source": app.state.model_bundle["source"],
        "model_alias": app.state.model_bundle["alias"],
        "model_version": app.state.model_bundle["version"],
        "run_id": app.state.model_bundle.get("run_id"),
    }


@app.post("/train")
async def train(
    smoke_test: bool = Query(False),
    local_data: Optional[str] = Query(None),
    bucket: Optional[str] = Query(None),
    prefix: Optional[str] = Query(None),
):
    args = []

    if smoke_test:
        args.append("--smoke-test")

    if local_data:
        args.extend(["--local-data", local_data])

    if bucket:
        args.extend(["--bucket", bucket])

    if prefix:
        args.extend(["--prefix", prefix])

    return run_script(TRAIN_SCRIPT, args)


@app.post("/promote")
async def promote(req: PromoteRequest):
    args = []
    if req.version is not None:
        args.extend(["--version", str(req.version)])
    return run_script(PROMOTE_SCRIPT, args)


@app.post("/rollback")
async def rollback(req: RollbackRequest):
    args = []
    if req.version is not None:
        args.extend(["--version", str(req.version)])
    return run_script(ROLLBACK_SCRIPT, args)


@app.post("/retrain-trigger")
async def retrain_trigger():
    return run_script(RETRAIN_TRIGGER_SCRIPT)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")



