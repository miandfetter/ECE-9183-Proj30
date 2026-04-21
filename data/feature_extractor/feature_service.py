#!/usr/bin/env python3

import base64
import io
import json
import os
import time
from datetime import datetime

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI(
    title="Online Feature Extraction Service",
    description="Computes log-mel spectrograms from raw audio for ASR inference",
    version="1.0.0",
)

_METRICS = {
    "total_requests": 0,
    "successful"    : 0,
    "failed"        : 0,
    "total_latency" : 0.0,
    "start_time"    : datetime.utcnow().isoformat(),
}



class FeatureResponse(BaseModel):
    request_id   : str
    features_b64 : str        # base64-encoded float32 array
    dtype        : str
    shape        : list       # [N_MELS, MAX_FRAMES]
    duration_sec : float
    sample_rate  : int
    latency_ms   : float
    timestamp    : str


class HealthResponse(BaseModel):
    status    : str
    version   : str
    timestamp : str



@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status"   : "healthy",
        "version"  : "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics")
def metrics():
    avg_latency = (
        _METRICS["total_latency"] / _METRICS["total_requests"]
        if _METRICS["total_requests"] > 0 else 0.0
    )
    return {
        **_METRICS,
        "avg_latency_ms": round(avg_latency, 2),
    }


@app.post("/extract", response_model=FeatureResponse)
async def extract(audio: UploadFile = File(...)):
    
    t0 = time.time()
    _METRICS["total_requests"] += 1
    request_id = f"feat_{_METRICS['total_requests']:06d}"

    
    if audio.content_type not in ("audio/wav", "audio/wave",
                                   "audio/x-wav", "application/octet-stream"):
        _METRICS["failed"] += 1
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {audio.content_type}. Expected audio/wav"
        )

    try:
        audio_bytes = await audio.read()

        if len(audio_bytes) == 0:
            raise ValueError("Empty audio file")

        result = extract_features(audio_bytes, normalize=True)
        serialized = features_to_dict(result)

        latency_ms = (time.time() - t0) * 1000
        _METRICS["successful"] += 1
        _METRICS["total_latency"] += latency_ms

        return {
            "request_id"  : request_id,
            "latency_ms"  : round(latency_ms, 2),
            "timestamp"   : datetime.utcnow().isoformat(),
            **serialized,
        }

    except Exception as e:
        _METRICS["failed"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-and-forward")
async def extract_and_forward(
    audio: UploadFile = File(...),
    model_endpoint: str = os.environ.get("MODEL_ENDPOINT", "")
):
    
    import requests as req

    
    audio_bytes = await audio.read()
    result = extract_features(audio_bytes, normalize=True)
    serialized = features_to_dict(result)

    if not model_endpoint:
        
        return JSONResponse({
            "status" : "features_only",
            "message": "Set MODEL_ENDPOINT to enable forwarding",
            **serialized,
        })

    
    try:
        features_array = result["features"]
        payload = {
            "inputs": [{
                "name" : "mel_features",
                "shape": list(features_array.shape),
                "dtype": "FP32",
                "data" : features_array.flatten().tolist(),
            }]
        }
        resp = req.post(
            f"{model_endpoint}/infer",
            json=payload,
            timeout=30
        )
        return JSONResponse({
            "status"          : "forwarded",
            "feature_shape"   : list(features_array.shape),
            "model_response"  : resp.json(),
        })

    except Exception as e:
        return JSONResponse({
            "status" : "forward_failed",
            "error"  : str(e),
            **serialized,
        }, status_code=502)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting feature extraction service on port {port}")
    print(f"Model endpoint: {os.environ.get('MODEL_ENDPOINT', 'not set')}")
    uvicorn.run(app, host="0.0.0.0", port=port)
