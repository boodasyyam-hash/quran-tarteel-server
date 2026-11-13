FROM python:3.10-slim

# تثبيت ffmpeg لمعالجة ملفات الصوت
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# تحميل نموذج ترتيل مسبقاً أثناء البناء لتسريع تشغيل السيرفر
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('OdyAsh/faster-whisper-base-ar-quran', device='cpu', compute_type='int8')"

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
