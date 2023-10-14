import argparse
import subprocess
import tempfile
from pathlib import Path

from summarize import summarize
from transcribe import to_srt, transcribe


def extract_audio(video: Path, out: Path):
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(out)],
                   check=True, capture_output=True)


VIDEO_EXT = {".mp4", ".mkv", ".mov", ".m4a", ".mp3", ".wav", ".webm"}


def find_recordings(paths):
    for p in paths:
        if p.is_dir():
            yield from sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
        else:
            yield p


def main():
    p = argparse.ArgumentParser(description="lecture recordings -> transcripts + notes")
    p.add_argument("recordings", type=Path, nargs="+", help="files or folders")
    p.add_argument("--model", default="small", help="whisper model size")
    p.add_argument("--no-summary", action="store_true")
    p.add_argument("--redo", action="store_true", help="re-process recordings that already have output")
    args = p.parse_args()

    for rec in find_recordings(args.recordings):
        out_dir = Path("output") / rec.stem
        if (out_dir / "transcript.txt").exists() and not args.redo:
            print(f"skip {rec.name} (already done)")
            continue
        print(f"== {rec.name}")
        process(rec, out_dir, args)


def process(recording, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.wav"
        print("extracting audio...")
        extract_audio(recording, audio)
        print("transcribing...")
        segments = transcribe(audio, args.model)

    (out_dir / "transcript.srt").write_text(to_srt(segments))
    (out_dir / "transcript.txt").write_text("\n".join(t for _, _, t in segments))
    print(f"transcript: {len(segments)} segments")

    if not args.no_summary:
        print("summarizing...")
        (out_dir / "notes.md").write_text(summarize(segments))
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
