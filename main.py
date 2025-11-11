from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import torch

app = FastAPI()

# Загружаем модель при старте сервера
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Сохраняем временный файл
    with open("temp_audio", "wb") as buffer:
        buffer.write(await file.read())
    
    # Транскрибируем
    result = pipe("temp_audio")
    return {"text": result["text"]}