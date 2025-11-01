from pathlib import Path

SCRIPTS = Path("generated_scripts")
TO_RENDER = Path("to_render")  # your make_video.py can watch this
TO_RENDER.mkdir(exist_ok=True)

DURATION = 30.0

def to_srt_text(body, duration_sec=DURATION):
    return f"1\n00:00:00,000 --> 00:00:{int(duration_sec):02d},000\n{body}\n"

for script in SCRIPTS.glob("*.txt"):
    raw = script.read_text(encoding="utf-8").strip()
    # drop title line if present
    lines = [l for l in raw.splitlines() if l.strip()]
    if len(lines) > 1:
        body = "\n".join(lines[1:-1]).strip() if lines[-1].startswith("— Source") else "\n".join(lines[1:]).strip()
    else:
        body = raw

    stem = script.stem
    (TO_RENDER / f"{stem}.txt").write_text(body, encoding="utf-8")
    (TO_RENDER / f"{stem}.srt").write_text(to_srt_text(body), encoding="utf-8")

print("Ready in:", TO_RENDER)

