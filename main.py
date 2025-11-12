from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import torch
import logging
import os
import uuid

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("Начинаю загрузку модели Whisper...")

# Загружаем модель при старте сервера
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

logger.info("Модель успешно загружена!")
logger.info(f"Используется устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    logger.info(f"Получен запрос на транскрибацию: {file.filename}")
    
    # Создаем уникальное имя временного файла
    temp_filename = f"temp_audio_{uuid.uuid4().hex[:8]}.webm"
    
    try:
        # Сохраняем временный файл
        logger.info(f"Сохраняю временный файл: {temp_filename}")
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            logger.info(f"Файл сохранен, размер: {len(content)} байт")
        
        # Транскрибируем
        logger.info("Начинаю транскрибацию...")
        result = pipe(temp_filename)
        logger.info("Транскрибация завершена успешно!")
        
        return {"text": result["text"]}
    
    except Exception as e:
        logger.error(f"Ошибка при транскрибации: {str(e)}")
        return {"error": str(e)}
    
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            logger.info(f"Временный файл {temp_filename} удален")

@app.get("/")
async def root():
    logger.info("Получен запрос на корневой эндпоинт")
    return {"message": "Транскрибатор работает!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}