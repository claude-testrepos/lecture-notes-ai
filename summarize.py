import openai

CHUNK_PROMPT = """You are summarizing part of a university lecture transcript.
Write concise study notes in Markdown: key concepts, definitions, formulas, and any
examples the lecturer works through. Do not invent content.

Transcript:
{text}"""

MERGE_PROMPT = """Combine these partial lecture notes into one well-organized Markdown document
with headings. Remove repetition. End with a "Possible exam questions" section.

{notes}"""


def _chunks(segments, max_chars=12000):
    buf = ""
    for _, _, text in segments:
        if len(buf) + len(text) > max_chars:
            yield buf
            buf = ""
        buf += text + " "
    if buf:
        yield buf


def summarize(segments, model="gpt-4"):
    def ask(prompt):
        r = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return r["choices"][0]["message"]["content"]

    parts = [ask(CHUNK_PROMPT.format(text=c)) for c in _chunks(segments)]
    return parts[0] if len(parts) == 1 else ask(MERGE_PROMPT.format(notes="\n\n---\n\n".join(parts)))
