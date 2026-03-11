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
