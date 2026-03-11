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
