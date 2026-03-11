import pytest
pytest.importorskip("torch")
from unittest.mock import MagicMock, patch
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
