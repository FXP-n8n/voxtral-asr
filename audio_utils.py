import os
import tempfile
import wave
from dataclasses import dataclass


@dataclass
class AudioChunk:
    path: str
    start: float  # seconds
    end: float    # seconds


def get_duration_seconds(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
    return n_frames / sample_rate


def split_audio_chunks(audio_path: str, chunk_seconds: int = 30) -> list[AudioChunk]:
    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    total_seconds = n_frames / sample_rate
    chunk_frames = chunk_seconds * sample_rate
    chunks = []
    offset_frames = 0

    while offset_frames < n_frames:
        end_frames = min(offset_frames + chunk_frames, n_frames)
        n_chunk_frames = int(end_frames - offset_frames)

        # Extract chunk data
        start_byte = int(offset_frames * n_channels * sample_width)
        end_byte = int(end_frames * n_channels * sample_width)
        chunk_data = audio_data[start_byte:end_byte]

        # Write chunk to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        with wave.open(tmp.name, "wb") as wf_out:
            wf_out.setnchannels(n_channels)
            wf_out.setsampwidth(sample_width)
            wf_out.setframerate(sample_rate)
            wf_out.writeframes(chunk_data)

        chunks.append(AudioChunk(
            path=tmp.name,
            start=offset_frames / sample_rate,
            end=end_frames / sample_rate,
        ))
        offset_frames = end_frames

    return chunks
