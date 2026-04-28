import os
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


CFG = {
    "model_name": "facebook/bart-large-cnn",
    "lr": 2e-4,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "max_input_len": 1024,
    "max_target_len": 128,
    "batch_size": 2,
    "accumulate_grad_batches": 8,
    "max_epochs": 3,
    "precision": "bf16-mixed",

    "mlflow_experiment": "bart-lora-meeting-summarization",
    "registered_model_name": "bart-meeting-summarizer",

    "baseline_rougeL": 0.18,
    "gate_tolerance": 0.02,

    "adapter_output_dir": "./bart_lora_adapter",
    "checkpoint_dir": "./checkpoints",

    # IMPORTANT: keep your working MeetingBank adapter
    "meetingbank_run_id": "242d296541bf40c5ae76d8d405d3b092",
    "meetingbank_artifact_path": "bart_lora_adapter",
}


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        verify=False,
    )


def load_bucket_dataset(bucket_name: str, prefix: str = "raw/") -> Dataset:
    s3 = get_s3_client()

    resp = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter="/",
    )

    meeting_prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    print(f"[data] Found {len(meeting_prefixes)} meeting folders under {prefix}")

    records = []

    for mp in meeting_prefixes:
        try:
            t_obj = s3.get_object(Bucket=bucket_name, Key=f"{mp}transcript.json")
            m_obj = s3.get_object(Bucket=bucket_name, Key=f"{mp}metadata.json")

            t_data = json.loads(t_obj["Body"].read())
            m_data = json.loads(m_obj["Body"].read())

            if isinstance(t_data, dict):
                transcript_text = t_data.get("transcript", "")
            elif isinstance(t_data, list):
                transcript_text = "\n".join(str(x) for x in t_data)
            else:
                transcript_text = str(t_data)

            summary_text = m_data.get("summary", "")

            if not transcript_text or not summary_text:
                print(f"[data] SKIP {mp}: empty transcript or summary")
                continue

            records.append(
                {
                    "id": str(m_data.get("uid", m_data.get("id", mp))),
                    "transcript": transcript_text,
                    "summary": summary_text,
                    "source": mp.rstrip("/").split("/")[-1],
                }
            )

        except Exception as e:
            print(f"[data] SKIP {mp}: {e}")

    print(f"[data] Loaded {len(records)} records from bucket")
    return Dataset.from_list(records)


def load_local_dataset(local_path: str) -> Dataset:
    with open(local_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Local data must be a JSON list")

    cleaned = []

    for i, r in enumerate(records):
        transcript = r.get("transcript", "")
        summary = r.get("summary", "")

        if not transcript or not summary:
            print(f"[data] SKIP local record {i}: empty transcript or summary")
            continue

        cleaned.append(
            {
                "id": str(r.get("id", i)),
                "transcript": transcript,
                "summary": summary,
                "source": r.get("source", "local"),
            }
        )

    print(f"[data] Loaded {len(cleaned)} local records from {local_path}")
    return Dataset.from_list(cleaned)

from mlflow.tracking import MlflowClient

client = MlflowClient("http://127.0.0.1:8002")

client.set_registered_model_alias(
    name="bart-meeting-summarizer",
    alias="production",
    version="1"
)
def tokenize(ds: Dataset, tokenizer) -> Dataset:
    def _tok(batch):
        # IMPORTANT:
        # Keep raw transcript format because your working adapter was trained this way.
        inputs = tokenizer(
            batch["transcript"],
            padding="max_length",
            truncation=True,
            max_length=CFG["max_input_len"],
        )

        targets = tokenizer(
            batch["summary"],
            padding="max_length",
            truncation=True,
            max_length=CFG["max_target_len"],
        )

        labels = [
            [
                token if token != tokenizer.pad_token_id else -100
                for token in label
            ]
            for label in targets["input_ids"]
        ]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

    tokenized = ds.map(
        _tok,
        batched=True,
        remove_columns=ds.column_names,
        desc="tokenize",
    )

    tokenized.set_format("torch")
    return tokenized


def load_base_adapter(cfg):
    run_id = cfg.get("meetingbank_run_id")
    artifact_path = cfg.get("meetingbank_artifact_path")

    if not run_id:
        print("[model] No MeetingBank run_id found. Cold start from BART.")
        return None

    try:
        print(f"[model] Downloading MeetingBank adapter from run {run_id}")
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )

        config_path = os.path.join(local_path, "adapter_config.json")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            valid_keys = {
                "peft_type",
                "task_type",
                "r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
                "bias",
                "inference_mode",
                "modules_to_save",
                "fan_in_fan_out",
                "init_lora_weights",
                "layers_to_transform",
                "layers_pattern",
                "rank_pattern",
                "alpha_pattern",
            }

            cleaned = {k: v for k, v in config.items() if k in valid_keys}

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2)

            print("[model] Adapter config patched successfully")

        print(f"[model] Warm-start adapter loaded from {local_path}")
        return local_path

    except Exception as e:
        print(f"[model] Failed to load MeetingBank adapter: {e}")
        print("[model] Falling back to cold-start BART")
        return None


class BARTLoRAFineTuner(L.LightningModule):
    def __init__(self, cfg, adapter_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.lr = cfg["lr"]

        base = BartForConditionalGeneration.from_pretrained(
            cfg["model_name"],
            torch_dtype=torch.float32,
        )

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                base,
                adapter_path,
                is_trainable=True,
            )
            print(f"[model] Loaded warm-start adapter from {adapter_path}")

        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=["q_proj", "v_proj"],
            )

            self.model = get_peft_model(base, lora_cfg)
            print("[model] Cold start from pretrained BART")

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
        if torch.cuda.is_available():
            print("[optim] Using bitsandbytes Adam8bit")
            return bnb.optim.Adam8bit(self.model.parameters(), lr=self.lr)

        print("[optim] Using torch AdamW on CPU")
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


def evaluate_rouge(model, loader, tokenizer, device, max_new_tokens):
    rouge = evaluate.load("rouge")
    model.eval()
    model.to(device)

    predictions = []
    references = []

    for batch in loader:
        with torch.no_grad():
            ids = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                num_beams=1,
                no_repeat_ngram_size=4,
            )

        preds = tokenizer.batch_decode(ids, skip_special_tokens=True)

        labels = [
            [token for token in label if token != -100]
            for label in batch["labels"].tolist()
        ]

        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(preds)
        references.extend(refs)

    scores = rouge.compute(
        predictions=predictions,
        references=references,
    )

    return {k: float(v) for k, v in scores.items()}, predictions, references


def get_current_staging_metrics(cfg):
    client = mlflow.MlflowClient()

    try:
        mv = client.get_model_version_by_alias(
            cfg["registered_model_name"],
            "staging",
        )

        run = client.get_run(mv.run_id)
        rouge_l = float(run.data.metrics.get("rougeL", 0))

        print(f"[gate] Current staging v{mv.version} rougeL={rouge_l:.4f}")
        return rouge_l

    except Exception:
        print(f"[gate] No staging model. Using baseline {cfg['baseline_rougeL']}")
        return cfg["baseline_rougeL"]


def get_production_rougeL(cfg):
    client = mlflow.MlflowClient()

    try:
        mv = client.get_model_version_by_alias(
            cfg["registered_model_name"],
            "production",
        )

        run = client.get_run(mv.run_id)
        rouge_l = float(run.data.metrics.get("rougeL", 0))

        print(f"[promote] Current production v{mv.version} rougeL={rouge_l:.4f}")
        return rouge_l, int(mv.version)

    except Exception:
        print("[promote] No production model found")
        return 0.0, None


def trigger_deploy(model_version: int):
    deploy_url = os.environ.get("INFERENCE_SERVER_URL")

    if not deploy_url:
        print("[deploy] INFERENCE_SERVER_URL not set. Skipping reload.")
        return

    try:
        import urllib.request

        req = urllib.request.Request(
            f"{deploy_url}/reload",
            data=json.dumps({"alias": "production"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[deploy] Inference server reload status: {resp.status}")

    except Exception as e:
        print(f"[deploy] Reload failed but training succeeded: {e}")


def gate_and_register(metrics, cfg, adapter_dir, mlf_logger):
    client = mlflow.MlflowClient()
    run_id = mlf_logger.run_id

    current_best = get_current_staging_metrics(cfg)
    threshold = current_best - cfg["gate_tolerance"]
    passed = metrics["rougeL"] >= threshold

    client.log_param(run_id, "gate_current_staging_rougeL", current_best)
    client.log_param(run_id, "gate_threshold", threshold)
    client.log_param(run_id, "gate_passed", passed)

    print(f"[gate] Current staging rougeL: {current_best:.4f}")
    print(f"[gate] Candidate rougeL: {metrics['rougeL']:.4f}")
    print(f"[gate] Threshold: {threshold:.4f}")

    if not passed:
        print("[gate] FAILED. Model not registered.")
        return False

    print("[gate] PASSED. Registering adapter.")

    client.log_artifacts(
        run_id=run_id,
        local_dir=adapter_dir,
        artifact_path="adapter",
    )

    run = client.get_run(run_id)
    source = run.info.artifact_uri.rstrip("/") + "/adapter"

    mv = client.create_model_version(
        name=cfg["registered_model_name"],
        source=source,
        run_id=run_id,
    )

    print(f"[gate] Registered model version v{mv.version}")

    client.set_registered_model_alias(
        name=cfg["registered_model_name"],
        alias="staging",
        version=mv.version,
    )

    prod_rouge_l, prod_version = get_production_rougeL(cfg)

    if metrics["rougeL"] > prod_rouge_l:
        client.set_registered_model_alias(
            name=cfg["registered_model_name"],
            alias="production",
            version=mv.version,
        )

        client.log_param(run_id, "promoted_to_production", True)

        print(
            f"[promote] Production updated to v{mv.version} "
            f"because {metrics['rougeL']:.4f} > {prod_rouge_l:.4f}"
        )

        trigger_deploy(int(mv.version))

    else:
        client.log_param(run_id, "promoted_to_production", False)

        print(
            f"[promote] Not promoted. Candidate {metrics['rougeL']:.4f} "
            f"<= production {prod_rouge_l:.4f}"
        )

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bucket",
        default=os.environ.get("BUCKET_NAME"),
        help="Object storage bucket containing raw/meeting_XXX folders",
    )

    parser.add_argument(
        "--prefix",
        default=os.environ.get("DATA_PREFIX", "raw/"),
        help="Prefix inside bucket",
    )

    parser.add_argument(
        "--local-data",
        default=None,
        help="Local JSON file with transcript and summary records",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 epoch and skip registration",
    )

    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise SystemExit("ERROR: MLFLOW_TRACKING_URI is not set")

    if not args.local_data and not args.bucket:
        raise SystemExit("ERROR: pass --local-data or set BUCKET_NAME")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(CFG["mlflow_experiment"])

    mlf_logger = MLFlowLogger(
        experiment_name=CFG["mlflow_experiment"],
        tracking_uri=tracking_uri,
        run_name=f"train-{int(time.time())}",
        log_model=False,
    )

    print("\n=== Loading data ===")

    if args.local_data:
        ds = load_local_dataset(args.local_data)
    else:
        ds = load_bucket_dataset(args.bucket, prefix=args.prefix)

    if len(ds) == 0:
        raise SystemExit("ERROR: No training records found")

    print("\n=== Tokenizing ===")
    tokenizer = BartTokenizer.from_pretrained(CFG["model_name"])
    tokenized = tokenize(ds, tokenizer)

    cpu_mode = not torch.cuda.is_available()

    train_loader = DataLoader(
        tokenized,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=0 if cpu_mode else 2,
        pin_memory=not cpu_mode,
    )

    eval_loader = DataLoader(
        tokenized,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=0 if cpu_mode else 2,
        pin_memory=not cpu_mode,
    )

    print(f"[data] {len(tokenized)} records loaded")

    print("\n=== Loading model ===")
    adapter_path = load_base_adapter(CFG)

    mlf_logger.log_hyperparams(
        {
            **{
                k: v
                for k, v in CFG.items()
                if isinstance(v, (int, float, str, bool))
            },
            "warm_start": adapter_path is not None,
            "bucket": args.bucket,
            "prefix": args.prefix,
            "local_data": args.local_data,
            "smoke_test": args.smoke_test,
            "cuda_available": torch.cuda.is_available(),
        }
    )

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

    start = time.time()
    trainer.fit(model, train_loader, eval_loader)
    train_time = time.time() - start

    client = mlflow.MlflowClient()
    client.log_metric(mlf_logger.run_id, "train_time_seconds", train_time)

    print(f"[train] Completed in {train_time / 60:.2f} minutes")

    print("\n=== Saving adapter ===")

    adapter_dir = Path(CFG["adapter_output_dir"]) / f"run-{mlf_logger.run_id}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"[train] Adapter saved to {adapter_dir}")

    client.log_artifacts(
        run_id=mlf_logger.run_id,
        local_dir=str(adapter_dir),
        artifact_path="adapter",
    )

    print("\n=== Evaluating ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics, preds, refs = evaluate_rouge(
        model.model,
        eval_loader,
        tokenizer,
        device,
        max_new_tokens=CFG["max_target_len"],
    )

    print(f"[eval] {metrics}")

    for k, v in metrics.items():
        client.log_metric(mlf_logger.run_id, k, v)

    samples = "\n\n".join(
        f"--- example {i} ---\nPRED: {p[:300]}\nREF: {r[:300]}"
        for i, (p, r) in enumerate(zip(preds[:5], refs[:5]))
    )

    samples_path = adapter_dir / "sample_predictions.txt"
    samples_path.write_text(samples, encoding="utf-8")
    client.log_artifact(mlf_logger.run_id, str(samples_path))

    if args.smoke_test:
        print("[gate] Smoke test mode. Skipping gate/register.")
        exit_code = 0
    else:
        print("\n=== Gate and register ===")
        passed = gate_and_register(metrics, CFG, str(adapter_dir), mlf_logger)
        exit_code = 0 if passed else 1

    client.set_terminated(mlf_logger.run_id)

    print(f"\n[done] MLflow run_id: {mlf_logger.run_id}")
    print(f"[done] exit_code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
