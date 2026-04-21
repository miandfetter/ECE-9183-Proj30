from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()


class TranscriptItem(BaseModel):
    speaker: int
    speaker_name: str
    text: str
    time: str | None = None


class SummaryRequest(BaseModel):
    room_id: str
    transcript: List[TranscriptItem]


def convert_to_meetingbank_format(transcript: List[TranscriptItem]):
    segments = []

    for item in transcript:
        segments.append({
            "speaker": item.speaker,
            "nbest": [
                {
                    "text": item.text
                }
            ]
        })

    return {"segments": segments}


def generate_summary_with_model(model_input):
    print("MODEL INPUT RECEIVED:")
    print(model_input)

    parts = []
    for seg in model_input["segments"]:
        if seg.get("nbest") and len(seg["nbest"]) > 0:
            text = seg["nbest"][0]["text"].strip()
            if text:
                parts.append(text)

    joined_text = " ".join(parts)
    return f"Meeting summary: {joined_text}"


@app.get("/")
def root():
    return {"message": "FastAPI summary service is running"}


@app.post("/summarize")
def summarize(data: SummaryRequest):
    transcript = data.transcript
    model_input = convert_to_meetingbank_format(transcript)

    summary = generate_summary_with_model(model_input)

    return {
        "room_id": data.room_id,
        "summary": summary,
        "model_input": model_input
    }