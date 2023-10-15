# lecture2notes

Turns recorded lectures into a transcript and Markdown study notes. Uses faster-whisper locally for speech-to-text and GPT-4 for summarizing.

## Requirements

ffmpeg on PATH, Python 3.10+, `OPENAI_API_KEY` set.

## Usage

```
pip install -r requirements.txt
python main.py lecture-week5.mp4 --model medium
python main.py recordings/          # everything in a folder
```
