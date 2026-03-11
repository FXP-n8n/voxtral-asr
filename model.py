import asyncio
import logging
import os
from functools import partial

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

import config
from audio_utils import split_audio_chunks, AudioChunk
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
