import logging
import os
import pathlib
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

import config
from jobs import JobStatus, JobStore, TranscriptResult
from srt import segments_to_srt

logger = logging.getLogger(__name__)

app = FastAPI(title="Voxtral ASR")
job_store = JobStore(max_queue=config.MAX_QUEUE_SIZE)
model_manager = None  # injected at startup

_static_dir = pathlib.Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    html = _static_dir / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return PlainTextResponse("Frontend not found", status_code=404)


@app.get("/api/config")
async def get_config():
    return {
        "model_variant": config.MODEL_VARIANT,
        "languages": config.SUPPORTED_LANGUAGES,
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "max_audio_minutes": config.MAX_AUDIO_SECONDS // 60,
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    timestamps: bool = Form(False),
):
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in config.SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {sorted(config.SUPPORTED_FORMATS)}")

    data = await file.read()
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.MAX_FILE_SIZE_MB} MB limit")

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    safe_name = f"{int(time.time() * 1000)}_{pathlib.Path(file.filename or 'audio').name}"
    dest = os.path.join(config.UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f_out:
        f_out.write(data)

    try:
        from audio_utils import get_duration_seconds
        duration = get_duration_seconds(dest)
    except Exception as e:
        os.unlink(dest)
        raise HTTPException(400, f"Invalid audio file: {e}")

    if duration > config.MAX_AUDIO_SECONDS:
        os.unlink(dest)
        raise HTTPException(400, f"Audio exceeds 30-minute limit ({duration:.0f}s)")

    try:
        job = job_store.create(
            filename=file.filename or safe_name,
            audio_path=dest,
            language=language if language != "auto" else None,
            timestamps=timestamps,
        )
    except OverflowError:
        os.unlink(dest)
        raise HTTPException(429, "Queue full. Try again later.")

    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    payload = {
        "id": job.id,
        "filename": job.filename,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "error": job.error,
    }
    if job.result:
        payload["text"] = job.result.text
    return payload


@app.get("/api/jobs/{job_id}/download")
async def download(job_id: str, format: str = "txt"):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETE or job.result is None:
        raise HTTPException(400, "Job not complete")

    if format == "srt":
        content = segments_to_srt(job.result.segments) or job.result.text
        return Response(
            content=content.encode(),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{job.filename}.srt"'},
        )
    return Response(
        content=job.result.text.encode(),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{job.filename}.txt"'},
    )
