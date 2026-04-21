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


@app.get("/")
def root():
    return {"message": "FastAPI summary service is running"}


@app.post("/summarize")
def summarize(data: SummaryRequest):
    transcript = data.transcript
    model_input = convert_to_meetingbank_format(transcript)

    full_text = " ".join([item.text for item in transcript])

    speakers = list(set([item.speaker_name for item in transcript]))

    summary = f"Meeting with {len(speakers)} participant(s). Key discussion: {full_text}"

    return {
        "room_id": data.room_id,
        "summary": summary,
        "model_input": model_input
    }