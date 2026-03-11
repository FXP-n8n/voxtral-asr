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
        from jobs import TranscriptResult, Segment
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
