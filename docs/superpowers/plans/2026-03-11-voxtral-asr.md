# Voxtral ASR Batch Transcription Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-hosted FastAPI web app for batch audio transcription using Mistral's Voxtral models with a drag-and-drop browser UI.

**Architecture:** Single Python process — FastAPI serves static HTML/JS frontend and a REST API; an asyncio job queue serializes transcription work; a ModelManager loads Voxtral once at startup via HuggingFace transformers on ROCm/CUDA/CPU.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, transformers≥4.54.0, mistral-common[audio]≥1.8.1, torch (ROCm), pydub, accelerate, python-multipart

---

## Chunk 1: Foundation — Config, Tests Scaffold, Requirements

### Task 1: Project scaffold and requirements

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `config.py`

- [ ] **Step 1: Write `requirements.txt`**

```
torch
transformers>=4.54.0
mistral-common[audio]>=1.8.1
accelerate
pydub
fastapi
uvicorn[standard]
python-multipart
pytest
pytest-asyncio
httpx
```

- [ ] **Step 2: Create `tests/__init__.py`** (empty file)

- [ ] **Step 3: Write failing tests for config**

File: `tests/test_config.py`

```python
import os
import pytest


def test_default_model_variant():
    os.environ.pop("MODEL_VARIANT", None)
    import importlib, config
    importlib.reload(config)
    assert config.MODEL_VARIANT == "mini"


def test_custom_model_variant():
    os.environ["MODEL_VARIANT"] = "small"
    import importlib, config
    importlib.reload(config)
    assert config.MODEL_VARIANT == "small"
    os.environ.pop("MODEL_VARIANT")


def test_default_upload_dir():
    os.environ.pop("UPLOAD_DIR", None)
    import importlib, config
    importlib.reload(config)
    assert config.UPLOAD_DIR == "./uploads"


def test_default_max_file_size():
    os.environ.pop("MAX_FILE_SIZE_MB", None)
    import importlib, config
    importlib.reload(config)
    assert config.MAX_FILE_SIZE_MB == 500


def test_default_host_port():
    import importlib, config
    importlib.reload(config)
    assert config.HOST == "0.0.0.0"
    assert config.PORT == 8000
```

- [ ] **Step 4: Run tests — verify they fail**

```
pytest tests/test_config.py -v
```
Expected: ImportError or AttributeError (config.py missing)

- [ ] **Step 5: Write `config.py`**

```python
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
```

- [ ] **Step 6: Run tests — verify they pass**

```
pytest tests/test_config.py -v
```
Expected: 5 PASSED

- [ ] **Step 7: Commit**

```bash
git add requirements.txt config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add config module with env-var driven settings"
```

---

## Chunk 2: Job Queue

### Task 2: Job dataclass and state machine

**Files:**
- Create: `jobs.py`
- Create: `tests/test_jobs.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_jobs.py`

```python
import asyncio
import pytest
from jobs import JobStore, Job, JobStatus


def test_create_job():
    store = JobStore()
    job = store.create("test.mp3")
    assert job.status == JobStatus.PENDING
    assert job.filename == "test.mp3"
    assert job.id in store.jobs


def test_get_job():
    store = JobStore()
    job = store.create("test.mp3")
    fetched = store.get(job.id)
    assert fetched is job


def test_get_missing_job_returns_none():
    store = JobStore()
    assert store.get("nonexistent") is None


def test_job_transitions_to_running():
    store = JobStore()
    job = store.create("test.mp3")
    store.set_running(job.id)
    assert store.get(job.id).status == JobStatus.RUNNING


def test_job_transitions_to_complete():
    from jobs import TranscriptResult, Segment
    store = JobStore()
    job = store.create("test.mp3")
    result = TranscriptResult(
        text="hello world",
        segments=[Segment(start=0.0, end=5.0, text="hello world")]
    )
    store.set_complete(job.id, result)
    j = store.get(job.id)
    assert j.status == JobStatus.COMPLETE
    assert j.result.text == "hello world"


def test_job_transitions_to_error():
    store = JobStore()
    job = store.create("test.mp3")
    store.set_error(job.id, "boom")
    j = store.get(job.id)
    assert j.status == JobStatus.ERROR
    assert j.error == "boom"


def test_queue_size_limit():
    store = JobStore(max_queue=2)
    store.create("a.mp3")
    store.create("b.mp3")
    with pytest.raises(OverflowError):
        store.create("c.mp3")


@pytest.mark.asyncio
async def test_worker_processes_jobs():
    from jobs import TranscriptResult, Segment
    store = JobStore()
    results = []

    async def fake_transcribe(audio_path, language, timestamps):
        results.append(audio_path)
        return TranscriptResult(text="hi", segments=[])

    job = store.create("x.mp3", audio_path="/tmp/x.mp3")
    task = asyncio.create_task(store.run_worker(fake_transcribe))
    await store.queue.join()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert store.get(job.id).status == JobStatus.COMPLETE
    assert "/tmp/x.mp3" in results
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_jobs.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `jobs.py`**

```python
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str
    segments: list[Segment]


@dataclass
class Job:
    id: str
    filename: str
    status: JobStatus
    audio_path: Optional[str]
    language: Optional[str]
    timestamps: bool
    created_at: datetime
    result: Optional[TranscriptResult] = None
    error: Optional[str] = None


class JobStore:
    def __init__(self, max_queue: int = 10):
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self._max_queue = max_queue

    def _pending_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING)

    def create(
        self,
        filename: str,
        audio_path: Optional[str] = None,
        language: Optional[str] = None,
        timestamps: bool = False,
    ) -> Job:
        if self._pending_count() >= self._max_queue:
            raise OverflowError("Queue full")
        job = Job(
            id=str(uuid.uuid4()),
            filename=filename,
            status=JobStatus.PENDING,
            audio_path=audio_path,
            language=language,
            timestamps=timestamps,
            created_at=datetime.now(timezone.utc),
        )
        self.jobs[job.id] = job
        self.queue.put_nowait(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        self.jobs[job_id].status = JobStatus.RUNNING

    def set_complete(self, job_id: str, result: TranscriptResult) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.COMPLETE
        job.result = result

    def set_error(self, job_id: str, error: str) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.ERROR
        job.error = error

    async def run_worker(self, transcribe_fn) -> None:
        """Consume jobs from queue indefinitely."""
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                self.queue.task_done()
                continue
            self.set_running(job_id)
            try:
                result = await transcribe_fn(
                    job.audio_path, job.language, job.timestamps
                )
                self.set_complete(job_id, result)
            except Exception as exc:
                self.set_error(job_id, str(exc))
            finally:
                self.queue.task_done()
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_jobs.py -v
```
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add jobs.py tests/test_jobs.py
git commit -m "feat: add job queue with state machine and async worker"
```

---

## Chunk 3: SRT Formatting and Audio Chunking Utilities

### Task 3: SRT formatter

**Files:**
- Create: `srt.py`
- Create: `tests/test_srt.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_srt.py`

```python
from srt import segments_to_srt, format_timestamp
from jobs import Segment


def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00,000"


def test_format_timestamp_full():
    assert format_timestamp(3723.456) == "01:02:03,456"


def test_single_segment():
    segs = [Segment(start=0.0, end=5.0, text="Hello world")]
    out = segments_to_srt(segs)
    assert "1\n" in out
    assert "00:00:00,000 --> 00:00:05,000" in out
    assert "Hello world" in out


def test_multiple_segments():
    segs = [
        Segment(start=0.0, end=5.0, text="First"),
        Segment(start=5.0, end=10.5, text="Second"),
    ]
    out = segments_to_srt(segs)
    assert "1\n" in out
    assert "2\n" in out
    assert "00:00:05,000 --> 00:00:10,500" in out


def test_empty_segments():
    assert segments_to_srt([]) == ""
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_srt.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `srt.py`**

```python
from jobs import Segment


def format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[Segment]) -> str:
    if not segments:
        return ""
    blocks = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(blocks)
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_srt.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add srt.py tests/test_srt.py
git commit -m "feat: add SRT formatter with timestamp conversion"
```

### Task 4: Audio chunking utility

**Files:**
- Create: `audio_utils.py`
- Create: `tests/test_audio_utils.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_audio_utils.py`

```python
import os
import tempfile
import wave
import struct
import pytest
from audio_utils import get_duration_seconds, split_audio_chunks, AudioChunk


def _make_wav(path: str, duration_s: float, sample_rate: int = 16000):
    """Create a minimal valid WAV file."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


def test_get_duration_seconds():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    _make_wav(path, 5.0)
    try:
        dur = get_duration_seconds(path)
        assert abs(dur - 5.0) < 0.1
    finally:
        os.unlink(path)


def test_split_audio_chunks_single():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    _make_wav(path, 10.0)
    try:
        chunks = split_audio_chunks(path, chunk_seconds=30)
        assert len(chunks) == 1
        assert chunks[0].start == 0.0
        assert abs(chunks[0].end - 10.0) < 0.2
    finally:
        os.unlink(path)
        for c in chunks:
            if os.path.exists(c.path):
                os.unlink(c.path)


def test_split_audio_chunks_multiple():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    _make_wav(path, 75.0)
    try:
        chunks = split_audio_chunks(path, chunk_seconds=30)
        assert len(chunks) == 3
        assert chunks[0].start == 0.0
        assert abs(chunks[1].start - 30.0) < 0.5
        assert abs(chunks[2].start - 60.0) < 0.5
    finally:
        os.unlink(path)
        for c in chunks:
            if os.path.exists(c.path):
                os.unlink(c.path)
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_audio_utils.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `audio_utils.py`**

```python
import os
import tempfile
from dataclasses import dataclass
from pydub import AudioSegment


@dataclass
class AudioChunk:
    path: str
    start: float  # seconds
    end: float    # seconds


def get_duration_seconds(audio_path: str) -> float:
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0


def split_audio_chunks(audio_path: str, chunk_seconds: int = 30) -> list[AudioChunk]:
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_ms = chunk_seconds * 1000
    chunks = []
    offset = 0
    while offset < total_ms:
        end_ms = min(offset + chunk_ms, total_ms)
        segment = audio[offset:end_ms]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        segment.export(tmp.name, format="wav")
        chunks.append(AudioChunk(
            path=tmp.name,
            start=offset / 1000.0,
            end=end_ms / 1000.0,
        ))
        offset = end_ms
    return chunks
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_audio_utils.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add audio_utils.py tests/test_audio_utils.py
git commit -m "feat: add audio chunking utility using pydub"
```

---

## Chunk 4: Model Manager

### Task 5: ModelManager with ROCm/CPU fallback

**Files:**
- Create: `model.py`
- Create: `tests/test_model.py`

Note: Tests mock the HuggingFace model — actual model loading requires GPU/large RAM and is tested manually.

- [ ] **Step 1: Write failing tests**

File: `tests/test_model.py`

```python
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from jobs import TranscriptResult, Segment


def test_transcript_result_has_text_and_segments():
    result = TranscriptResult(
        text="hello",
        segments=[Segment(start=0.0, end=1.0, text="hello")]
    )
    assert result.text == "hello"
    assert len(result.segments) == 1


@patch("model.AutoProcessor")
@patch("model.VoxtralForConditionalGeneration")
def test_model_manager_loads_mini(mock_model_cls, mock_proc_cls):
    mock_model_cls.from_pretrained.return_value = MagicMock()
    mock_proc_cls.from_pretrained.return_value = MagicMock()

    from model import ModelManager
    mgr = ModelManager(variant="mini")
    mgr.load()

    mock_model_cls.from_pretrained.assert_called_once()
    call_args = mock_model_cls.from_pretrained.call_args
    assert "Voxtral-Mini" in call_args[0][0]


@patch("model.AutoProcessor")
@patch("model.VoxtralForConditionalGeneration")
def test_model_manager_invalid_variant_raises(mock_model_cls, mock_proc_cls):
    from model import ModelManager
    with pytest.raises(ValueError, match="Unknown variant"):
        ModelManager(variant="giant")


@pytest.mark.asyncio
@patch("model.AutoProcessor")
@patch("model.VoxtralForConditionalGeneration")
async def test_transcribe_no_timestamps_returns_result(mock_model_cls, mock_proc_cls):
    import torch
    mock_proc = MagicMock()
    mock_proc_cls.from_pretrained.return_value = mock_proc
    mock_model = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model

    # Simulate generate returning token ids, processor decoding to text
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_proc.batch_decode.return_value = ["hello world"]
    mock_proc.apply_transcription_request.return_value = {
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[1]]),
    }

    from model import ModelManager
    mgr = ModelManager(variant="mini")
    mgr.load()

    import tempfile, wave, struct, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    n = 8000
    with wave.open(path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n}h", *([0]*n)))

    result = await mgr.transcribe(path, language=None, timestamps=False)
    os.unlink(path)

    assert isinstance(result, TranscriptResult)
    assert isinstance(result.text, str)
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_model.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `model.py`**

```python
import asyncio
import logging
import os
from functools import partial

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

import config
from audio_utils import split_audio_chunks, get_duration_seconds, AudioChunk
from jobs import TranscriptResult, Segment

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, variant: str = config.MODEL_VARIANT):
        if variant not in config.MODEL_IDS:
            raise ValueError(f"Unknown variant '{variant}'. Choose: {list(config.MODEL_IDS)}")
        self.variant = variant
        self.model_id = config.MODEL_IDS[variant]
        self.model = None
        self.processor = None
        self._device = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id} ...")
        rocm_available = torch.cuda.is_available()
        dtype = torch.float16

        if rocm_available:
            self._device = "cuda"
            kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        else:
            logger.warning("ROCm/CUDA not available — falling back to CPU (slow)")
            self._device = "cpu"
            kwargs = {"torch_dtype": torch.float32, "device_map": "cpu"}

        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info("Model loaded.")

    async def transcribe(
        self,
        audio_path: str,
        language: str | None,
        timestamps: bool,
    ) -> TranscriptResult:
        loop = asyncio.get_event_loop()
        fn = partial(self._transcribe_sync, audio_path, language, timestamps)
        return await loop.run_in_executor(None, fn)

    def _transcribe_sync(
        self,
        audio_path: str,
        language: str | None,
        timestamps: bool,
    ) -> TranscriptResult:
        if timestamps:
            return self._transcribe_chunked(audio_path, language)
        else:
            return self._transcribe_full(audio_path, language)

    def _transcribe_full(self, audio_path: str, language: str | None) -> TranscriptResult:
        lang = language if language and language != "auto" else None
        inputs = self.processor.apply_transcription_request(
            language=lang,
            audio=audio_path,
            model_id=self.model_id,
        )
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return TranscriptResult(text=text, segments=[])

    def _transcribe_chunked(self, audio_path: str, language: str | None) -> TranscriptResult:
        chunks: list[AudioChunk] = split_audio_chunks(audio_path, config.AUDIO_CHUNK_SECONDS)
        all_segments: list[Segment] = []
        all_text_parts: list[str] = []
        lang = language if language and language != "auto" else None

        try:
            for chunk in chunks:
                inputs = self.processor.apply_transcription_request(
                    language=lang,
                    audio=chunk.path,
                    model_id=self.model_id,
                )
                inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs)
                text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                all_text_parts.append(text)
                all_segments.append(Segment(start=chunk.start, end=chunk.end, text=text))
        finally:
            for chunk in chunks:
                if os.path.exists(chunk.path):
                    os.unlink(chunk.path)

        return TranscriptResult(text=" ".join(all_text_parts), segments=all_segments)
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_model.py -v
```
Expected: 4 PASSED (mocked model tests)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add ModelManager with ROCm/CPU fallback and chunked timestamp mode"
```

---

## Chunk 5: API Layer

### Task 6: FastAPI endpoints

**Files:**
- Create: `api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing integration tests**

File: `tests/test_api.py`

```python
import asyncio
import io
import struct
import wave
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock, AsyncMock


def _wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    n = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0]*n)))
    return buf.getvalue()


@pytest.fixture
def mock_model_manager():
    from jobs import TranscriptResult, Segment
    mgr = MagicMock()
    mgr.transcribe = AsyncMock(return_value=TranscriptResult(
        text="hello world",
        segments=[Segment(start=0.0, end=1.0, text="hello world")]
    ))
    return mgr


@pytest.mark.asyncio
async def test_get_config(mock_model_manager):
    with patch("api.model_manager", mock_model_manager):
        from api import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_variant" in data
    assert "languages" in data


@pytest.mark.asyncio
async def test_upload_returns_job_id(mock_model_manager):
    with patch("api.model_manager", mock_model_manager):
        from api import app, job_store
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            wav = _wav_bytes()
            resp = await client.post(
                "/api/transcribe",
                files={"file": ("test.wav", wav, "audio/wav")},
                data={"language": "en", "timestamps": "false"},
            )
    assert resp.status_code == 200
    assert "job_id" in resp.json()


@pytest.mark.asyncio
async def test_upload_unsupported_format_returns_400(mock_model_manager):
    with patch("api.model_manager", mock_model_manager):
        from api import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/transcribe",
                files={"file": ("test.txt", b"not audio", "text/plain")},
                data={"language": "en", "timestamps": "false"},
            )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_job_status(mock_model_manager):
    with patch("api.model_manager", mock_model_manager):
        from api import app, job_store
        # pre-create a job
        job = job_store.create("x.wav", audio_path="/tmp/x.wav")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/jobs/{job.id}")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("pending", "running", "complete", "error")


@pytest.mark.asyncio
async def test_get_missing_job_returns_404(mock_model_manager):
    with patch("api.model_manager", mock_model_manager):
        from api import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/jobs/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_download_txt(mock_model_manager):
    with patch("api.model_manager", mock_model_manager):
        from api import app, job_store
        from jobs import TranscriptResult, Segment, JobStatus
        job = job_store.create("y.wav")
        job_store.set_complete(job.id, TranscriptResult(
            text="test transcript",
            segments=[Segment(start=0.0, end=1.0, text="test transcript")]
        ))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/jobs/{job.id}/download?format=txt")
    assert resp.status_code == 200
    assert b"test transcript" in resp.content


@pytest.mark.asyncio
async def test_download_srt(mock_model_manager):
    with patch("api.model_manager", mock_model_manager):
        from api import app, job_store
        from jobs import TranscriptResult, Segment
        job = job_store.create("z.wav")
        job_store.set_complete(job.id, TranscriptResult(
            text="test transcript",
            segments=[Segment(start=0.0, end=1.0, text="test transcript")]
        ))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/jobs/{job.id}/download?format=srt")
    assert resp.status_code == 200
    assert b"-->" in resp.content
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_api.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `api.py`**

```python
import logging
import os
import pathlib
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

import config
from jobs import Job, JobStatus, JobStore, TranscriptResult
from model import ModelManager
from srt import segments_to_srt

logger = logging.getLogger(__name__)

app = FastAPI(title="Voxtral ASR")
job_store = JobStore(max_queue=config.MAX_QUEUE_SIZE)
model_manager: Optional[ModelManager] = None  # injected at startup


# --- Static files ---
_static_dir = pathlib.Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    html = _static_dir / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return PlainTextResponse("Frontend not found", status_code=404)


# --- Config ---
@app.get("/api/config")
async def get_config():
    return {
        "model_variant": config.MODEL_VARIANT,
        "languages": config.SUPPORTED_LANGUAGES,
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "max_audio_minutes": config.MAX_AUDIO_SECONDS // 60,
    }


# --- Upload & transcribe ---
@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    timestamps: bool = Form(False),
):
    # Validate extension
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in config.SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {config.SUPPORTED_FORMATS}")

    # Read and check size
    data = await file.read()
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.MAX_FILE_SIZE_MB} MB limit")

    # Save to uploads dir
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    safe_name = f"{int(time.time() * 1000)}_{pathlib.Path(file.filename or 'audio').name}"
    dest = os.path.join(config.UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f:
        f.write(data)

    # Validate audio and duration
    try:
        from audio_utils import get_duration_seconds
        duration = get_duration_seconds(dest)
    except Exception as e:
        os.unlink(dest)
        raise HTTPException(400, f"Invalid audio file: {e}")

    if duration > config.MAX_AUDIO_SECONDS:
        os.unlink(dest)
        raise HTTPException(400, f"Audio exceeds 30-minute limit ({duration:.0f}s)")

    # Enqueue
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


# --- Job status ---
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


# --- Download ---
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
    else:
        return Response(
            content=job.result.text.encode(),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{job.filename}.txt"'},
        )
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_api.py -v
```
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api.py
git commit -m "feat: add FastAPI endpoints for upload, job status, and transcript download"
```

---

## Chunk 6: Entry Point and Frontend

### Task 7: `main.py` entry point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write `main.py`**

```python
import asyncio
import logging
import os

import uvicorn

import config
from api import app, job_store
from model import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


async def _start_worker(model_mgr: ModelManager):
    await job_store.run_worker(model_mgr.transcribe)


@app.on_event("startup")
async def startup():
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    model_mgr = ModelManager(variant=config.MODEL_VARIANT)
    try:
        model_mgr.load()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise SystemExit(1)

    # patch model_manager into api module
    import api
    api.model_manager = model_mgr

    asyncio.create_task(_start_worker(model_mgr))
    logger.info("Worker started.")


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add main.py entry point with model load and worker startup"
```

### Task 8: Frontend `static/index.html`

**Files:**
- Create: `static/index.html`

- [ ] **Step 1: Write `static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Voxtral ASR</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 2rem; }
    h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 1.5rem; color: #f8fafc; }
    .card { background: #1e2130; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }

    /* Drop zone */
    #drop-zone {
      border: 2px dashed #4a5568; border-radius: 10px; padding: 2.5rem;
      text-align: center; cursor: pointer; transition: border-color .2s, background .2s;
    }
    #drop-zone.hover { border-color: #6366f1; background: #2d2f4a; }
    #drop-zone p { color: #94a3b8; }
    #file-input { display: none; }

    /* Options */
    .options { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; margin-top: 1rem; }
    select, button { padding: .5rem 1rem; border-radius: 8px; border: 1px solid #4a5568;
      background: #2d3348; color: #e2e8f0; font-size: .95rem; }
    button { cursor: pointer; background: #6366f1; border-color: #6366f1; font-weight: 600; }
    button:hover { background: #4f46e5; }
    button:disabled { background: #4a5568; border-color: #4a5568; cursor: not-allowed; }
    label.toggle { display: flex; align-items: center; gap: .5rem; font-size: .9rem; cursor: pointer; }

    /* Jobs */
    .job-card { background: #262b3e; border-radius: 8px; padding: 1rem; margin-bottom: .75rem; }
    .job-header { display: flex; justify-content: space-between; align-items: center; }
    .badge { padding: .2rem .6rem; border-radius: 99px; font-size: .75rem; font-weight: 700; text-transform: uppercase; }
    .badge.pending   { background: #374151; color: #9ca3af; }
    .badge.running   { background: #1e3a5f; color: #60a5fa; }
    .badge.complete  { background: #14532d; color: #4ade80; }
    .badge.error     { background: #4c1d1d; color: #f87171; }

    textarea { width: 100%; margin-top: .75rem; padding: .75rem; background: #0f1117;
      border: 1px solid #4a5568; border-radius: 8px; color: #e2e8f0; resize: vertical; min-height: 120px; font-family: monospace; font-size: .85rem; }
    .dl-row { display: flex; gap: .5rem; margin-top: .5rem; }
    .dl-btn { background: #1e3a5f; border-color: #1e3a5f; font-size: .82rem; padding: .35rem .75rem; }
    .copy-btn { background: #2d3348; border-color: #4a5568; font-size: .82rem; padding: .35rem .75rem; }
    #model-info { font-size: .8rem; color: #94a3b8; text-align: right; margin-bottom: .5rem; }
  </style>
</head>
<body>
  <h1>Voxtral ASR</h1>
  <div id="model-info">Loading config…</div>

  <div class="card">
    <div id="drop-zone">
      <p>Drop audio file here or <strong>click to browse</strong></p>
      <p style="font-size:.8rem;margin-top:.4rem;">MP3 · WAV · FLAC · OGG · max 30 min</p>
      <input type="file" id="file-input" accept=".mp3,.wav,.flac,.ogg" />
    </div>
    <div class="options">
      <select id="lang-select"><option value="auto">Auto-detect</option></select>
      <label class="toggle">
        <input type="checkbox" id="ts-toggle" /> Timestamps (SRT)
      </label>
      <button id="submit-btn" disabled>Transcribe</button>
    </div>
  </div>

  <div class="card">
    <h2 style="font-size:1rem;margin-bottom:.75rem;color:#94a3b8;">Jobs</h2>
    <div id="job-list"><p style="color:#4a5568;font-size:.9rem;">No jobs yet.</p></div>
  </div>

  <script>
    const API = '';
    const jobs = {};       // id -> { jobData, intervalId }
    let selectedFile = null;

    // --- Config ---
    fetch(`${API}/api/config`).then(r => r.json()).then(cfg => {
      document.getElementById('model-info').textContent = `Model: ${cfg.model_variant} · max ${cfg.max_audio_minutes} min`;
      const sel = document.getElementById('lang-select');
      cfg.languages.forEach(lang => {
        if (lang === 'auto') return;
        const opt = document.createElement('option');
        opt.value = lang; opt.textContent = lang.toUpperCase();
        sel.appendChild(opt);
      });
    });

    // --- Drop zone ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const submitBtn = document.getElementById('submit-btn');

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('hover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('hover'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropZone.classList.remove('hover');
      const f = e.dataTransfer.files[0];
      if (f) setFile(f);
    });

    function setFile(f) {
      selectedFile = f;
      dropZone.querySelector('p').textContent = `Selected: ${f.name}`;
      submitBtn.disabled = false;
    }

    // --- Submit ---
    submitBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      submitBtn.disabled = true;
      const fd = new FormData();
      fd.append('file', selectedFile);
      fd.append('language', document.getElementById('lang-select').value);
      fd.append('timestamps', document.getElementById('ts-toggle').checked ? 'true' : 'false');

      try {
        const res = await fetch(`${API}/api/transcribe`, { method: 'POST', body: fd });
        if (!res.ok) {
          const err = await res.json();
          alert(`Error: ${err.detail}`);
          submitBtn.disabled = false;
          return;
        }
        const { job_id } = await res.json();
        addJob(job_id, selectedFile.name);
        selectedFile = null;
        dropZone.querySelector('p').textContent = 'Drop audio file here or click to browse';
        fileInput.value = '';
      } catch (e) {
        alert('Upload failed: ' + e.message);
      }
      submitBtn.disabled = false;
    });

    // --- Jobs ---
    function addJob(id, filename) {
      const card = document.createElement('div');
      card.className = 'job-card'; card.id = `job-${id}`;
      card.innerHTML = `
        <div class="job-header">
          <span style="font-size:.9rem;font-weight:600;">${escHtml(filename)}</span>
          <span class="badge pending" id="badge-${id}">pending</span>
        </div>
        <div id="result-${id}"></div>`;
      const list = document.getElementById('job-list');
      if (list.querySelector('p')) list.innerHTML = '';
      list.prepend(card);

      const iid = setInterval(() => pollJob(id), 2000);
      jobs[id] = { intervalId: iid };
      pollJob(id);
    }

    async function pollJob(id) {
      try {
        const res = await fetch(`${API}/api/jobs/${id}`);
        if (!res.ok) return;
        const data = await res.json();
        updateJob(id, data);
        if (data.status === 'complete' || data.status === 'error') {
          clearInterval(jobs[id].intervalId);
        }
      } catch (_) {}
    }

    function updateJob(id, data) {
      const badge = document.getElementById(`badge-${id}`);
      if (badge) { badge.className = `badge ${data.status}`; badge.textContent = data.status; }

      const resultDiv = document.getElementById(`result-${id}`);
      if (!resultDiv) return;

      if (data.status === 'complete' && data.text) {
        resultDiv.innerHTML = `
          <textarea readonly>${escHtml(data.text)}</textarea>
          <div class="dl-row">
            <button class="copy-btn" onclick="navigator.clipboard.writeText(document.querySelector('#result-${id} textarea').value)">Copy</button>
            <button class="dl-btn" onclick="dlFile('${id}','txt')">Download .txt</button>
            <button class="dl-btn" onclick="dlFile('${id}','srt')">Download .srt</button>
          </div>`;
      } else if (data.status === 'error') {
        resultDiv.innerHTML = `<p style="color:#f87171;font-size:.85rem;margin-top:.5rem;">${escHtml(data.error || 'Unknown error')}</p>`;
      }
    }

    function dlFile(id, fmt) {
      window.open(`${API}/api/jobs/${id}/download?format=${fmt}`, '_blank');
    }

    function escHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }
  </script>
</body>
</html>
```

- [ ] **Step 2: Verify static dir structure**

```
ls static/
```
Expected: `index.html`

- [ ] **Step 3: Commit**

```bash
git add static/index.html main.py
git commit -m "feat: add frontend UI with drag-and-drop upload and job polling"
```

---

## Chunk 7: Integration Test and Final Wiring

### Task 9: Integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

File: `tests/test_integration.py`

```python
"""
Integration test: upload a real short WAV and poll until complete.
Requires pydub and a real (or mocked) transcription path.
Run with: pytest tests/test_integration.py -v
"""
import asyncio
import io
import struct
import wave
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from jobs import TranscriptResult, Segment


def _wav_bytes(duration_s: float = 2.0) -> bytes:
    n = int(duration_s * 16000)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n}h", *([0]*n)))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_full_transcription_flow():
    fake_result = TranscriptResult(
        text="this is a test",
        segments=[Segment(start=0.0, end=2.0, text="this is a test")]
    )

    with patch("api.model_manager") as mgr_mock:
        mgr_mock.transcribe = AsyncMock(return_value=fake_result)

        from api import app, job_store

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # 1. Upload
            wav = _wav_bytes()
            upload_resp = await client.post(
                "/api/transcribe",
                files={"file": ("speech.wav", wav, "audio/wav")},
                data={"language": "en", "timestamps": "false"},
            )
            assert upload_resp.status_code == 200
            job_id = upload_resp.json()["job_id"]

            # 2. Manually trigger worker (no background task in test context)
            job = job_store.get(job_id)
            assert job is not None
            # Simulate worker completing the job
            job_store.set_running(job_id)
            result = await mgr_mock.transcribe(job.audio_path, job.language, job.timestamps)
            job_store.set_complete(job_id, result)

            # 3. Poll status
            status_resp = await client.get(f"/api/jobs/{job_id}")
            assert status_resp.status_code == 200
            data = status_resp.json()
            assert data["status"] == "complete"
            assert "this is a test" in data["text"]

            # 4. Download txt
            dl_resp = await client.get(f"/api/jobs/{job_id}/download?format=txt")
            assert dl_resp.status_code == 200
            assert b"this is a test" in dl_resp.content

            # 5. Download srt (no segments in non-timestamp mode is fine — falls back to text)
            srt_resp = await client.get(f"/api/jobs/{job_id}/download?format=srt")
            assert srt_resp.status_code == 200
```

- [ ] **Step 2: Run integration test**

```
pytest tests/test_integration.py -v
```
Expected: 1 PASSED

- [ ] **Step 3: Run the full test suite**

```
pytest tests/ -v
```
Expected: All tests PASSED

- [ ] **Step 4: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test for transcription flow"
```

---

## Running the App

```bash
# Install deps (ROCm PyTorch must be installed separately per AMD docs)
pip install -r requirements.txt

# Run with Voxtral Mini (default)
python main.py

# Run with Small model
MODEL_VARIANT=small python main.py

# Open browser
# http://localhost:8000
```

Manual smoke-test checklist:
- [ ] Upload a short MP3 → job card appears, status transitions pending → running → complete
- [ ] Transcript text is non-empty
- [ ] Download .txt works
- [ ] Upload with Timestamps toggle → download .srt has `-->` markers
- [ ] Upload file >30 min → 400 error shown
- [ ] Upload unsupported format (e.g. .mp4) → 400 error shown
