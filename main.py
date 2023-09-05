import argparse
import subprocess
import tempfile
from pathlib import Path

from summarize import summarize
from transcribe import to_srt, transcribe


def extract_audio(video: Path, out: Path):
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(out)],
                   check=True, capture_output=True)


def main():
    p = argparse.ArgumentParser(description="lecture recording -> transcript + notes")
    p.add_argument("recording", type=Path)
    p.add_argument("--model", default="small", help="whisper model size")
    p.add_argument("--no-summary", action="store_true")
    args = p.parse_args()

    out_dir = Path("output") / args.recording.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.wav"
        print("extracting audio...")
        extract_audio(args.recording, audio)
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
