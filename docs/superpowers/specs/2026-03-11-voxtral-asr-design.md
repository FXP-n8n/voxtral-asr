# Voxtral ASR — Batch Transcription Web App

## Summary

A self-hosted web application for batch audio transcription using Mistral's open-weight Voxtral models. Users upload audio files through a browser UI and receive transcripts as plain text or timestamped SRT.

## Requirements

- Local GPU inference (no external API calls)
- Support Voxtral Mini (3B, bf16) and Voxtral Small (24B, 4-bit quantized), switchable via config
- Web UI with drag-and-drop upload
- Output: plain text and timestamped SRT
- Supported audio formats: MP3, WAV, FLAC, OGG
- Max audio duration: 30 minutes (Voxtral model limit)
- Language: auto-detect + manual override (en, es, fr, pt, hi, de, nl, it)

## Architecture

Single Python process: FastAPI backend serving a static HTML/JS frontend.

```
Browser  <──HTTP──>  FastAPI
                      ├── Static files (HTML/CSS/JS)
                      ├── Upload endpoint
                      ├── Job queue (in-memory, async)
                      └── Model Manager
                           └── Voxtral (transformers + bitsandbytes)
```

## Components

### 1. Config (`config.py`)

Environment-variable-driven configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VARIANT` | `mini` | `mini` or `small` |
| `UPLOAD_DIR` | `./uploads` | Temp storage for uploaded audio |
| `MAX_FILE_SIZE_MB` | `500` | Max upload size in MB |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

### 2. Model Manager (`model.py`)

- Loads model once at startup based on `MODEL_VARIANT`
- Target hardware: AMD Strix Halo APU (RDNA 3.5 iGPU, gfx1151) with 64GB unified LPDDR5X, using ROCm + PyTorch
- Mini: `VoxtralForConditionalGeneration.from_pretrained("mistralai/Voxtral-Mini-3B-2507", torch_dtype=torch.float16, device_map="auto")`
- Small: same class, `torch_dtype=torch.float16, device_map="auto"` — fits in 64GB unified memory (~48GB in fp16)
- Falls back to CPU if ROCm is unavailable (`device_map="cpu"` with float32)
- `transcribe(audio_path: str, language: str | None, timestamps: bool) -> TranscriptResult`
- `TranscriptResult`: dataclass with `text: str`, `segments: list[Segment]` (each has `start`, `end`, `text`)
- Uses `AutoProcessor.apply_transcription_request()` for input preparation

### 3. Job Queue (`jobs.py`)

- In-memory dict of `Job` objects (dataclass: id, status, filename, created_at, result, error)
- `asyncio.Queue` with a single worker coroutine consuming jobs sequentially
- Status: `pending` → `running` → `complete` | `error`
- Temp files cleaned up after transcription completes

### 4. API Layer (`api.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve index.html |
| `/api/transcribe` | POST | Upload audio file + options, returns job ID |
| `/api/jobs/{id}` | GET | Job status + result if complete |
| `/api/jobs/{id}/download` | GET | Download transcript as .txt or .srt (query param `format`) |
| `/api/config` | GET | Current model variant and supported languages |

### 5. Frontend (`static/index.html`)

Single HTML file with embedded CSS and JS:

- **Upload zone**: drag-and-drop or click-to-browse, file type validation
- **Options bar**: language selector dropdown, timestamps toggle
- **Job list**: cards showing filename, status badge (pending/running/complete/error), elapsed time
- **Transcript viewer**: text area with the result, copy button
- **Download buttons**: .txt and .srt
- Polls `/api/jobs/{id}` every 2 seconds while a job is running

## Dependencies

```
torch
transformers>=4.54.0
mistral-common[audio]>=1.8.1
accelerate
pydub
fastapi
uvicorn
python-multipart
```

## File Structure

```
ASR/
├── config.py
├── model.py
├── jobs.py
├── api.py
├── main.py          # entry point: uvicorn
├── requirements.txt
├── static/
│   └── index.html
└── uploads/         # created at runtime
```

## Error Handling

- File too large → 413 response
- Unsupported format → 400 response
- Transcription failure → job status `error` with message
- Model loading failure → startup crash with clear error message

## Timestamp Strategy

The Hugging Face transformers implementation of Voxtral does **not** support word-level or segment-level timestamps natively (this is only available via the Mistral hosted API). For SRT output, we use a chunk-based approach:

1. Split audio into fixed-duration chunks (e.g., 30-second segments) using `pydub` or `torchaudio`
2. Transcribe each chunk independently
3. Map each chunk's text to its time range (chunk_index * chunk_duration)
4. Combine into SRT format

This gives sentence/paragraph-level timestamps aligned to chunk boundaries. Not word-level, but sufficient for subtitle use cases.

**Dependency addition**: `pydub` (for audio chunking) and `ffmpeg` (system dependency for audio decoding).

## Operational Constraints

- **Queue limit**: Max 10 pending jobs. Additional uploads are rejected with 429.
- **Job retention**: Completed/errored jobs are evicted from memory after 1 hour. Temp files deleted immediately on completion or error.
- **Audio validation**: After upload, probe the file with `pydub.AudioSegment` to verify it's valid audio. Reject with 400 if it fails.
- **Duration enforcement**: Probe audio duration server-side; reject files over 30 minutes with 400.
- **Access model**: Single-user, no authentication. The spec intentionally omits auth since this is a local tool.
- **Memory estimates**: Mini fp16 ~6-7 GB, Small fp16 ~48 GB (fits in 64GB unified memory on Strix Halo).

## Entry Point (`main.py`)

Starts uvicorn with the FastAPI app, creates upload directory if missing, triggers model loading.

## Testing Strategy

- Unit tests for config parsing, job state machine, SRT formatting, audio chunking
- Integration test: upload a short audio file, poll until complete, verify transcript is non-empty
- Manual test: upload various formats and languages through the web UI
