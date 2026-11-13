import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel

app = FastAPI()

# تحميل نموذج ترتيل القرآني مضغوطاً (int8) ليناسب الباقة المجانية
MODEL_ID = "OdyAsh/faster-whisper-base-ar-quran"
model = WhisperModel(MODEL_ID, device="cpu", compute_type="int8")

@app.get("/")
def health_check():
    return {"status": "ready", "model": MODEL_ID}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        segments, _ = model.transcribe(
            tmp_path,
            language="ar",
            beam_size=1,
            vad_filter=True
        )
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        return {"text": text}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
