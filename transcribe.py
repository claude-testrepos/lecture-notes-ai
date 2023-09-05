from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(audio: Path, model_size="small", device="auto"):
    model = WhisperModel(model_size, device=device, compute_type="int8")
    segments, info = model.transcribe(str(audio), vad_filter=True, beam_size=5)
    print(f"detected language {info.language} ({info.language_probability:.0%})")
    return [(s.start, s.end, s.text.strip()) for s in segments]


def to_srt(segments) -> str:
    def ts(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02}:{int(m):02}:{s:06.3f}".replace(".", ",")

    return "\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{text}\n" for i, (a, b, text) in enumerate(segments, 1))
