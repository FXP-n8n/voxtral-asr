import os

MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "mini")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

SUPPORTED_LANGUAGES = ["auto", "en", "es", "fr", "pt", "hi", "de", "nl", "it"]
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg"}
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", str(4 * 60 * 60)))  # default 4 hours
MAX_QUEUE_SIZE = 10
JOB_RETENTION_SECONDS = 3600  # 1 hour
# Max new tokens per chunk — 15s of speech is ~150 words ~200 tokens; 512 is a safe ceiling
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
AUDIO_CHUNK_SECONDS = int(os.getenv("AUDIO_CHUNK_SECONDS", "15"))

MODEL_IDS = {
    "mini": "mistralai/Voxtral-Mini-3B-2507",
}
