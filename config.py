import os

MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "mini")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

SUPPORTED_LANGUAGES = ["auto", "en", "es", "fr", "pt", "hi", "de", "nl", "it"]
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg"}
MAX_AUDIO_SECONDS = 30 * 60  # 30 minutes
MAX_QUEUE_SIZE = 10
JOB_RETENTION_SECONDS = 3600  # 1 hour
AUDIO_CHUNK_SECONDS = 30

MODEL_IDS = {
    "mini": "mistralai/Voxtral-Mini-3B-2507",
    "small": "mistralai/Voxtral-Small-24B-2507",
}
