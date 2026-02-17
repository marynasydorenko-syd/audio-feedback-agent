from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import subprocess, os, uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = WhisperModel("base", device="cpu")

def run_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    path = f"{UPLOAD_DIR}/{file_id}.wav"

    with open(path, "wb") as f:
        f.write(await file.read())

    segments, _ = model.transcribe(path)
    transcription = " ".join(seg.text for seg in segments)

    prompt = f"""
You are an English teacher giving feedback.

Student transcription:
{transcription}

Return:
1. Natural content response (1–2 sentences)
2. Brief positive comment
3. 2–3 corrections only

Format:
Original sentence with underlined mistake
Correct sentence
Short explanation in brackets

Tone: friendly and human
"""

    feedback = run_llm(prompt)

    return {
        "transcription": transcription,
        "feedback": feedback
    }
