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
