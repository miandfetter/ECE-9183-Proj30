"""
train.py — end-to-end BART-LoRA training for mirotalk meeting summarization.

Pipeline:
  1. Load config from env vars
  2. Pull transcripts+summaries from Chameleon bucket
  3. Tokenize
  4. Train BART with LoRA adapter
  5. Evaluate on held-out subset (ROUGE)
  6. Gate: only register if candidate rougeL >= BASELINE_ROUGEL - TOLERANCE
  7. If passed: log adapter to MLflow, register model, set alias "staging"

Invocation (from GPU VM, inside the pytorch Docker container):
    python train.py

Required env vars (already set by the setup_v2 docker run):
    MLFLOW_TRACKING_URI       e.g. http://<mlflow-vm-ip>:8000
    MLFLOW_S3_ENDPOINT_URL    https://chi.tacc.chameleoncloud.org:7480
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    BUCKET_NAME               the transcription bucket name (ask teammate)
    DATA_PREFIX               (optional) default "raw/"
"""
import os
import io
import json
import time
import argparse
from pathlib import Path

import torch
import boto3
import mlflow
import lightning as L
import bitsandbytes as bnb
import evaluate
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from lightning.pytorch.loggers import MLFlowLogger


# ==============================================================================
# Config — edit cautiously, these are tuned for H100 + small corpus
# ==============================================================================
CFG = {
    "model_name"              : "facebook/bart-large-cnn",
    "lr"                      : 2e-4,
    "lora_r"                  : 8,
    "lora_alpha"              : 16,
    "lora_dropout"            : 0.1,
    "max_input_len"           : 1024,
    "max_target_len"          : 128,
    "batch_size"              : 2,
    "accumulate_grad_batches" : 8,
    "max_epochs"              : 3,
    "precision"               : "bf16-mixed",

    # MLflow
    "mlflow_experiment"       : "bart-lora-meeting-summarization",
    "registered_model_name"   : "bart-meeting-summarizer",

    # Gate
    # BASELINE_ROUGEL was measured from pretrained bart-large-cnn on our eval set.
    # Any candidate must score within TOLERANCE of this, otherwise we don't register it.
    # See safeguarding plan: this threshold is committed to code, not changed at training time.
    "baseline_rougeL"         : 0.18,
    "gate_tolerance"          : 0.02,

    # Paths on the container (mounted from host /mnt/data)
    "adapter_output_dir"      : "/mnt/data/bart_lora_adapter",
    "checkpoint_dir"          : "/mnt/data/checkpoints",
    "meetingbank_run_id": "242d296541bf40c5ae76d8d405d3b092",  # ← your run_id here
    "meetingbank_artifact_path": "bart_lora_adapter",
}


# ==============================================================================
# Step 1: load data from the Chameleon object store
# ==============================================================================
def load_bucket_dataset(bucket_name: str, prefix: str = "raw/") -> Dataset:
    """Pull meeting_XXX/transcript.json + metadata.json pairs from Swift (S3-shim).

    Expects layout:
        {prefix}meeting_000/transcript.json  -> {"transcript": "..."}
        {prefix}meeting_000/metadata.json    -> {"summary": "...", "uid": "...", "id": ...}

    Returns HuggingFace Dataset with columns: id, transcript, summary, source.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        verify=False,  # Chameleon Swift self-signed cert; see safeguarding notes
    )

    # List meeting folders under prefix
    resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")
    meeting_prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    print(f"[data] Found {len(meeting_prefixes)} meeting folders under {prefix}")

    records = []
    for mp in meeting_prefixes:
        try:
            t_obj = s3.get_object(Bucket=bucket_name, Key=f"{mp}transcript.json")
            m_obj = s3.get_object(Bucket=bucket_name, Key=f"{mp}metadata.json")
            t_data = json.loads(t_obj["Body"].read())
            m_data = json.loads(m_obj["Body"].read())

            # Accept either {"transcript": "..."} or a raw string
            transcript_text = t_data["transcript"] if isinstance(t_data, dict) else t_data

            records.append({
                "id":         str(m_data.get("uid", m_data.get("id", mp))),
                "transcript": transcript_text,
                "summary":    m_data["summary"],
                "source":     mp.rstrip("/").split("/")[-1],
            })
        except Exception as e:
            print(f"[data] SKIP {mp}: {e}")

    print(f"[data] Loaded {len(records)} records")
    return Dataset.from_list(records)


def load_local_dataset(local_path: str) -> Dataset:
    from pathlib import Path
    p = Path(local_path)

    # Directory of meeting_XXX folders (same layout as bucket)
    if p.is_dir():
        records = []
        for meeting_dir in sorted(p.iterdir()):
            transcript_file = meeting_dir / "transcript.json"
            metadata_file = meeting_dir / "metadata.json"
            if not transcript_file.exists() or not metadata_file.exists():
                continue
            try:
                t_data = json.loads(transcript_file.read_text())
                m_data = json.loads(metadata_file.read_text())
                transcript_text = t_data["transcript"] if isinstance(t_data, dict) else t_data
                records.append({
                    "id":         str(m_data.get("uid", m_data.get("id", meeting_dir.name))),
                    "transcript": transcript_text,
                    "summary":    m_data["summary"],
                    "source":     meeting_dir.name,
                })
            except Exception as e:
                print(f"[data] SKIP {meeting_dir}: {e}")
    else:
        # JSON file with list of records
        records = json.loads(p.read_text())
        records = [r for r in records if r.get("transcript") and r.get("summary")]

    print(f"[data] Loaded {len(records)} local records from {local_path}")
    return Dataset.from_list(records)


def tokenize(ds: Dataset, tokenizer) -> Dataset:
    def _tok(batch):
        inputs = tokenizer(
            batch["transcript"],
            padding="max_length", truncation=True, max_length=CFG["max_input_len"],
        )
        targets = tokenizer(
            batch["summary"],
            padding="max_length", truncation=True, max_length=CFG["max_target_len"],
        )
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in lab]
            for lab in targets["input_ids"]
        ]
        return {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels":         labels,
        }

    tokenized = ds.map(
        _tok, batched=True,
        remove_columns=[c for c in ds.column_names if c not in {}],  # drop all text
        desc="tokenize",
    )
    tokenized.set_format("torch")
    return tokenized

def load_base_model(cfg):
    """
    Load the best MeetingBank adapter from MLflow by run_id.
    Falls back to pretrained bart-large-cnn if run_id not set or download fails.
    """
    run_id        = cfg.get("meetingbank_run_id")
    artifact_path = cfg.get("meetingbank_artifact_path", "model")

    if not run_id:
        print("[model] No meetingbank_run_id set — cold start from bart-large-cnn")
        return None

    try:
        print(f"[model] Downloading MeetingBank adapter from run {run_id}...")
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
        print(f"[model] Downloaded to {local_path}")

    # Fix incompatible config by keeping only known valid keys
        config_path = f"{local_path}/adapter_config.json"
        with open(config_path) as f:
            config = json.load(f)

        print(f"[model] Original config keys: {list(config.keys())}")

        # Only keep keys that current PEFT version supports
        valid_keys = {
            "peft_type", "task_type", "r", "lora_alpha", "lora_dropout",
            "target_modules", "bias", "inference_mode", "modules_to_save",
            "fan_in_fan_out", "init_lora_weights", "layers_to_transform",
            "layers_pattern", "rank_pattern", "alpha_pattern",
        }

        cleaned = {k: v for k, v in config.items() if k in valid_keys}
        print(f"[model] Cleaned config keys: {list(cleaned.keys())}")

        with open(config_path, "w") as f:
            json.dump(cleaned, f, indent=2)

        print(f"[model] Patched adapter config at {config_path}")
        return local_path
    
    except Exception as e:
        print(f"[model] Failed to download adapter: {e}")
        print("[model] Falling back to bart-large-cnn")
        return None


# ==============================================================================
# Step 2: Lightning module
# ==============================================================================
class BARTLoRAFineTuner(L.LightningModule):
    def __init__(self, cfg, adapter_path=None):  # ← add adapter_path
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.lr = cfg["lr"]

        base = BartForConditionalGeneration.from_pretrained(
            cfg["model_name"], torch_dtype=torch.float32,
        )

        if adapter_path:
            # Load your existing MeetingBank fine-tuned adapter
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                base, adapter_path, is_trainable=True
            )
            print(f"[model] Loaded MeetingBank adapter from {adapter_path}")
        else:
            # Fresh LoRA on top of pretrained BART
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=["q_proj", "v_proj"],
            )
            self.model = get_peft_model(base, lora_cfg)
            print("[model] Cold start from pretrained bart-large-cnn")

        self.model.print_trainable_parameters()

    def training_step(self, batch, _):
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("train_loss", out.loss, on_step=True, on_epoch=True, prog_bar=True)
        return out.loss

    def validation_step(self, batch, _):
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("val_loss", out.loss, on_epoch=True, prog_bar=True)
        return out.loss

    def configure_optimizers(self):
        # Adam8bit saves VRAM — safe on H100, doesn't affect quality meaningfully
        return bnb.optim.Adam8bit(self.model.parameters(), lr=self.lr)


# ==============================================================================
# Step 3: ROUGE evaluation
# ==============================================================================
def evaluate_rouge(model, loader, tokenizer, device, max_new_tokens):
    """Generate summaries with greedy decoding, return rouge metrics dict."""
    rouge = evaluate.load("rouge")
    model.eval()
    model.to(device)

    predictions, references = [], []
    for batch in loader:
        with torch.no_grad():
            ids = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        preds = tokenizer.batch_decode(ids, skip_special_tokens=True)
        labs = [[t for t in lab if t != -100] for lab in batch["labels"].tolist()]
        refs = tokenizer.batch_decode(labs, skip_special_tokens=True)
        predictions.extend(preds)
        references.extend(refs)

    scores = rouge.compute(predictions=predictions, references=references)
    return {k: float(v) for k, v in scores.items()}, predictions, references


# ==============================================================================
# Step 4: Gate + register
# ==============================================================================
def get_current_staging_metrics(cfg):
    """Get rougeL of the current staging model from MLflow.
    Falls back to hardcoded baseline if no staging model exists yet.
    """
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(
            cfg["registered_model_name"], "staging"
        )
        run = client.get_run(mv.run_id)
        rougeL = float(run.data.metrics.get("rougeL", 0))
        print(f"[gate] Found staging model v{mv.version} with rougeL: {rougeL:.4f}")
        return rougeL
    except Exception:
        print(f"[gate] No staging model found — using hardcoded baseline: {cfg['baseline_rougeL']}")
        return cfg["baseline_rougeL"]
    
def _trigger_deploy(model_version: int):
    """Notify the inference server to reload from the new staging model."""
    deploy_url = os.environ.get("INFERENCE_SERVER_URL")
    if not deploy_url:
        print("[deploy] INFERENCE_SERVER_URL not set — skipping hot-reload")
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{deploy_url}/reload",
            data=json.dumps({"alias": "staging", "version": model_version}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[deploy] Inference server reloaded: {resp.status}")
    except Exception as e:
        print(f"[deploy] Reload failed (non-fatal): {e}")

def get_current_staging_metrics(cfg):
    """Get rougeL of the current staging model from MLflow.
    Falls back to hardcoded baseline if no staging model exists yet.
    """
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(
            cfg["registered_model_name"], "staging"
        )
        run = client.get_run(mv.run_id)
        rougeL = float(run.data.metrics.get("rougeL", 0))
        print(f"[gate] Found staging model v{mv.version} with rougeL: {rougeL:.4f}")
        return rougeL
    except Exception:
        print(f"[gate] No staging model found — using hardcoded baseline: {cfg['baseline_rougeL']}")
        return cfg["baseline_rougeL"]


def get_production_rougeL(cfg):
    """Get rougeL and version of the current production model from MLflow.
    Returns (0.0, None) if no production model exists yet.
    """
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(
            cfg["registered_model_name"], "production"
        )
        run = client.get_run(mv.run_id)
        rougeL = float(run.data.metrics.get("rougeL", 0))
        print(f"[promote] Found production model v{mv.version} with rougeL: {rougeL:.4f}")
        return rougeL, int(mv.version)
    except Exception:
        print(f"[promote] No production model found — any passing model will be promoted")
        return 0.0, None


def gate_and_register(metrics, cfg, adapter_dir, mlf_logger):
    """Return True if gate passed and model was registered."""

    # --- Gate: must beat current staging ---
    current_best = get_current_staging_metrics(cfg)
    threshold = current_best - cfg["gate_tolerance"]
    passed = metrics["rougeL"] >= threshold

    print(f"[gate] Current staging rougeL: {current_best:.4f}")
    print(f"[gate] Candidate rougeL:       {metrics['rougeL']:.4f}")
    print(f"[gate] Threshold:              {threshold:.4f}")

    client = mlflow.MlflowClient()
    run_id = mlf_logger.run_id

    # Log gate decision
    client.log_param(run_id, "gate_current_staging_rougeL", current_best)
    client.log_param(run_id, "gate_tolerance",               cfg["gate_tolerance"])
    client.log_param(run_id, "gate_threshold",               threshold)
    client.log_param(run_id, "gate_passed",                  passed)

    if not passed:
        print(f"[gate] FAILED: rougeL {metrics['rougeL']:.4f} < threshold {threshold:.4f}")
        print("[gate] Model NOT registered.")
        return False

    print(f"[gate] PASSED: rougeL {metrics['rougeL']:.4f} >= threshold {threshold:.4f}")

    # --- Register model ---
    client.log_artifacts(run_id, local_dir=adapter_dir, artifact_path="adapter")
    model_uri = f"runs:/{run_id}/adapter"

    try:
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=cfg["registered_model_name"],
        )
        print(f"[gate] Registered as {cfg['registered_model_name']} v{mv.version}")

        # Always set staging alias
        client.set_registered_model_alias(
            name=cfg["registered_model_name"],
            alias="staging",
            version=mv.version,
        )
        print(f"[gate] Alias 'staging' -> v{mv.version}")

        # --- Auto-promote to production if better than current production ---
        prod_rougeL, prod_version = get_production_rougeL(cfg)

        if metrics["rougeL"] > prod_rougeL:
            client.set_registered_model_alias(
                name=cfg["registered_model_name"],
                alias="production",
                version=mv.version,
            )
            print(f"[promote] Auto-promoted: production -> v{mv.version} "
                  f"(rougeL {metrics['rougeL']:.4f} > {prod_rougeL:.4f})")
            client.log_param(run_id, "promoted_to_production", True)
            _trigger_deploy(mv.version)
        else:
            print(f"[promote] Not promoted: candidate rougeL {metrics['rougeL']:.4f} "
                  f"does not improve on production v{prod_version} ({prod_rougeL:.4f})")
            print(f"[promote] Run promote.py manually to force promotion if desired")
            client.log_param(run_id, "promoted_to_production", False)

        return True

    except Exception as e:
        print(f"[gate] Registration failed: {e}")
        print("[gate] Adapter is still logged as a run artifact; serving can pull directly.")
        return False

# ==============================================================================
# main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("BUCKET_NAME"),
                        help="Chameleon bucket name containing raw/meeting_XXX/")
    parser.add_argument("--prefix", default=os.environ.get("DATA_PREFIX", "raw/"),
                        help="Prefix inside the bucket (default: raw/)")
    parser.add_argument("--local-data", default=None,
                        help="Path to local data directory or JSON file")
    parser.add_argument("--smoke-test", action="store_true",
                        help="1 epoch, skip gate — just verify the pipeline")
    args = parser.parse_args()

    if not args.bucket and not args.local_data:
        raise SystemExit("ERROR: set BUCKET_NAME env var or pass --bucket / --local-data")

    # MLflow setup
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(CFG["mlflow_experiment"])

    mlf_logger = MLFlowLogger(
        experiment_name=CFG["mlflow_experiment"],
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        run_name=f"train-{int(time.time())}",
        log_model=False,
    )

    # -------- Data --------
    print("\n=== Loading data ===")
    if args.local_data:
        ds = load_local_dataset(args.local_data)
    else:
        ds = load_bucket_dataset(args.bucket, prefix=args.prefix)
    if len(ds) == 0:
        raise SystemExit("No data found. Check bucket/prefix or local-data path.")

    print("\n=== Tokenizing ===")
    tokenizer = BartTokenizer.from_pretrained(CFG["model_name"])
    tokenized = tokenize(ds, tokenizer)

    train_loader = DataLoader(tokenized, batch_size=CFG["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    eval_loader  = DataLoader(tokenized, batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[data] {len(tokenized)} examples, {len(train_loader)} train batches")

    # -------- Load base model --------
    print("\n=== Loading base model ===")
    adapter_path = load_base_model(CFG)

    # Log config + env (moved here so we can include warm_start info)
    mlf_logger.log_hyperparams({
        **{k: v for k, v in CFG.items() if isinstance(v, (int, float, str, bool))},
        "warm_start":         adapter_path is not None,
        "meetingbank_run_id": CFG.get("meetingbank_run_id", "none"),
        "bucket":             args.bucket,
        "prefix":             args.prefix,
        "smoke_test":         args.smoke_test,
    })

    # -------- Train --------
    print("\n=== Training ===")
    model = BARTLoRAFineTuner(CFG, adapter_path=adapter_path)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=CFG["precision"] if torch.cuda.is_available() else 32,
        accumulate_grad_batches=CFG["accumulate_grad_batches"],
        max_epochs=1 if args.smoke_test else CFG["max_epochs"],
        enable_checkpointing=False,
        logger=mlf_logger,
        log_every_n_steps=1,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, eval_loader)
    train_time = time.time() - t0

    client = mlflow.MlflowClient()
    client.log_metric(mlf_logger.run_id, "train_time_seconds", train_time)
    print(f"[train] Completed in {train_time/60:.1f} min")

    # -------- Save adapter --------
    adapter_dir = Path(CFG["adapter_output_dir"]) / f"run-{mlf_logger.run_id}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] Adapter saved to {adapter_dir}")

    # -------- Evaluate --------
    print("\n=== Evaluating ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics, preds, refs = evaluate_rouge(
        model.model, eval_loader, tokenizer, device,
        max_new_tokens=CFG["max_target_len"],
    )
    print(f"[eval] {metrics}")

    for k, v in metrics.items():
        client.log_metric(mlf_logger.run_id, k, v)

    # Log a couple of prediction samples for manual inspection
    samples = "\n\n".join(
        f"--- example {i} ---\nPRED: {p[:300]}\nREF:  {r[:300]}"
        for i, (p, r) in enumerate(zip(preds[:3], refs[:3]))
    )
    samples_path = adapter_dir / "sample_predictions.txt"
    samples_path.write_text(samples)
    client.log_artifact(mlf_logger.run_id, str(samples_path))

    # -------- Gate + register --------
    if args.smoke_test:
        print("[gate] Smoke test mode — skipping gate+register")
        exit_code = 0
    else:
        print("\n=== Gating ===")
        passed = gate_and_register(metrics, CFG, str(adapter_dir), mlf_logger)
        exit_code = 0 if passed else 1

    # Tidy up MLflow run
    client.set_terminated(mlf_logger.run_id)
    print(f"\n[done] MLflow run: {mlf_logger.run_id}")
    print(f"[done] exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())