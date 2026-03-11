"""
Integration test: upload a short WAV, manually drive the job through completion,
verify transcript is retrievable and downloadable.
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

            # 2. Manually drive job to complete (no background worker in test)
            job = job_store.get(job_id)
            assert job is not None
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

            # 5. Download srt
            srt_resp = await client.get(f"/api/jobs/{job_id}/download?format=srt")
            assert srt_resp.status_code == 200
