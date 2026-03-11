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
