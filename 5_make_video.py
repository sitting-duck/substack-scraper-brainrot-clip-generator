#!/usr/bin/env python3
"""
make_video.py — Batch 30-second Shorts Generator + Optional Telegram Push
(with debug printouts)
"""

import os, re, json, time, shutil, tempfile, subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import requests
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip, ColorClip, vfx, ImageClip
)
from moviepy.audio.AudioClip import AudioClip, CompositeAudioClip

# --- Pillow 10+ compatibility shim ---
from PIL import Image, ImageDraw, ImageFont
try:
    Resampling = Image.Resampling
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Resampling.LANCZOS
    if not hasattr(Image, "BILINEAR"):
        Image.BILINEAR = Resampling.BILINEAR
    if not hasattr(Image, "BICUBIC"):
        Image.BICUBIC = Resampling.BICUBIC
except Exception:
    pass
# -------------------------------------

# ---------- Small logger ----------
def log(msg: str):
    print(f"[make_video] {msg}", flush=True)

# ---------- Config ----------
load_dotenv()

IN_DIR  = Path(os.getenv("IN_DIR", "to_render"))
OUT_DIR = Path(os.getenv("OUT_DIR", "rendered"))
DONE_DIR = Path(os.getenv("DONE_DIR", "done"))
FAILED_DIR = Path(os.getenv("FAILED_DIR", "failed"))
for p in [IN_DIR, OUT_DIR, DONE_DIR, FAILED_DIR]:
    p.mkdir(exist_ok=True)

DURATION = float(os.getenv("VIDEO_DURATION_SEC", "30"))
FRAME_W  = int(os.getenv("FRAME_W", "1080"))
FRAME_H  = int(os.getenv("FRAME_H", "1920"))
FPS      = int(os.getenv("FPS", "30"))

PEXELS_API_KEY        = os.getenv("PEXELS_API_KEY", "").strip()
PEXELS_QUERY_FALLBACK = os.getenv("PEXELS_QUERY_FALLBACK", "abstract bokeh city night")

EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")
USE_EDGE_TTS   = os.getenv("USE_EDGE_TTS", "true").lower() == "true"

ENABLE_CAPTIONS     = os.getenv("ENABLE_CAPTIONS", "false").lower() == "true"
SEND_TELEGRAM       = os.getenv("SEND_TELEGRAM", "false").lower() == "true"
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TARGET_CHAT         = [s.strip() for s in os.getenv("TARGET_CHAT", "").split(",") if s.strip()]
SUBSCRIBERS_FILE    = os.getenv("SUBSCRIBERS_FILE", "subscribers.json")
SEND_COOLDOWN_SEC   = float(os.getenv("SEND_COOLDOWN_SEC", "1.5"))

log(f"CFG: DURATION={DURATION}s, SIZE={FRAME_W}x{FRAME_H}@{FPS}fps "
    f"| PEXELS={'on' if PEXELS_API_KEY else 'off'} "
    f"| CAPTIONS={'on' if ENABLE_CAPTIONS else 'off'} "
    f"| TELEGRAM={'on' if (SEND_TELEGRAM and TELEGRAM_BOT_TOKEN) else 'off'} "
    f"| USE_EDGE_TTS={'on' if USE_EDGE_TTS else 'off'}")

# ---------- Helpers ----------
def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")[:140]

def read_targets() -> List[str]:
    targets = TARGET_CHAT.copy()
    if Path(SUBSCRIBERS_FILE).exists():
        try:
            data = json.loads(Path(SUBSCRIBERS_FILE).read_text())
            if isinstance(data, list):
                targets += [str(x) for x in data]
        except Exception as e:
            log(f"read_targets: failed to read {SUBSCRIBERS_FILE}: {e}")
    seen, uniq = set(), []
    for t in targets:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def find_scripts() -> List[Path]:
    return sorted(IN_DIR.glob("*.txt"))

def stem_pair(p: Path) -> Tuple[Path, Optional[Path]]:
    srt = p.with_suffix(".srt")
    return p, (srt if srt.exists() else None)

def keyword_guess(text: str, max_len: int = 50) -> str:
    words = [w.strip(",.!?;:()[]\"'").lower() for w in text.split()]
    words = [w for w in words if len(w) >= 4][:8]
    return " ".join(words[:6])[:max_len] if words else PEXELS_QUERY_FALLBACK

# ---------- Pexels ----------
def fetch_pexels_clips(query: str, target_sec: float) -> List[VideoFileClip]:
    if not PEXELS_API_KEY:
        log("PEXELS disabled (no API key).")
        return []
    headers = {"Authorization": PEXELS_API_KEY}
    url = "https://api.pexels.com/videos/search"
    params = {"query": query, "per_page": 10, "orientation": "portrait", "size": "medium"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        count = len(data.get('videos', []))
        print(f"[Pexels] query='{query}' returned {count} videos")
    except Exception as e:
        log(f"PEXELS request failed: {e}")
        return []

    clips = []
    tmpdir = Path(tempfile.mkdtemp(prefix="pex_"))
    for vid in data.get("videos", []):
        files = vid.get("video_files", [])
        portrait = [f for f in files if f.get("height", 0) > f.get("width", 0)]
        portrait.sort(key=lambda f: f.get("height", 0), reverse=True)
        if not portrait and files:
            portrait = files
        if portrait:
            link = portrait[0].get("link")
            try:
                vr = requests.get(link, timeout=30)
                if vr.status_code == 200:
                    dest = tmpdir / f"{vid.get('id','v')}.mp4"
                    dest.write_bytes(vr.content)
                    c = VideoFileClip(str(dest)).without_audio()
                    log(f"PEXELS clip: {dest.name} size={c.w}x{c.h}, dur={getattr(c,'duration',None):.2f}s, fps={getattr(c,'fps',None)}")
                    clips.append(c)
            except Exception as e:
                log(f"download/open failed for {link}: {e}")
        if sum([c.duration for c in clips]) >= target_sec + 3:
            break
    return clips

def fallback_bg(duration: float) -> VideoFileClip:
    base = (
        ColorClip(size=(FRAME_W, FRAME_H), color=(35, 35, 45))
        .set_duration(duration)
        .set_fps(FPS)
    )
    log(f"Fallback BG: size={FRAME_W}x{FRAME_H}, dur={duration}s, fps={FPS}")
    return base.fx(vfx.colorx, 1.05)

def assemble_broll(query: str, duration: float) -> VideoFileClip:
    clips = fetch_pexels_clips(query, duration)
    if not clips:
        log("No PEXELS clips, using fallback.")
        return fallback_bg(duration)
    normalized = []
    for c in clips:
        c = c.resize(height=FRAME_H)
        if c.w != FRAME_W:
            x1 = max(0, (c.w - FRAME_W)//2)
            c = c.crop(x1=x1, y1=0, x2=x1+FRAME_W, y2=FRAME_H)
        normalized.append(c.set_fps(FPS))
        log(f"Normalized clip: size={c.w}x{c.h}, dur={c.duration:.2f}s, fps={FPS}")
    if not normalized:
        log("Normalization yielded no clips; using fallback.")
        return fallback_bg(duration)
    chain, acc, i = [], 0.0, 0
    while acc < duration and normalized:
        piece = normalized[i % len(normalized)]
        take = min(piece.duration, max(1.0, duration - acc))
        seg = piece.subclip(0, take)
        chain.append(seg)
        acc += seg.duration
        i += 1
    out = concatenate_videoclips(chain, method="compose")
    log(f"B-roll assembled: total_dur={out.duration:.2f}s, size={out.w}x{out.h}, fps={getattr(out,'fps',None)}")
    return out

# ---------- TTS ----------
def tts_edge_cli(text: str, out_wav: Path) -> bool:
    try:
        import shutil as sh
        if sh.which("edge-tts") is None:
            return False
        cmd = ["edge-tts", "--voice", EDGE_TTS_VOICE, "--text", text, "--write-media", str(out_wav)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 0
        log(f"edge-tts: returncode={r.returncode}, wrote={ok}, size={out_wav.stat().st_size if out_wav.exists() else 0}")
        if not ok:
            log(f"edge-tts stderr: {r.stderr[:200] if r.stderr else ''}")
        return ok
    except Exception as e:
        log(f"edge-tts failed: {e}")
        return False

def tts_mac_say(text: str, out_aiff: Path) -> bool:
    try:
        r = subprocess.run(["say", "-v", "Samantha", "-o", str(out_aiff), text], capture_output=True, text=True)
        ok = (r.returncode == 0 and out_aiff.exists() and out_aiff.stat().st_size > 0)
        log(f"mac say: returncode={r.returncode}, wrote={ok}, size={out_aiff.stat().st_size if out_aiff.exists() else 0}")
        if not ok:
            log(f"mac say stderr: {r.stderr[:200] if r.stderr else ''}")
        return ok
    except Exception as e:
        log(f"mac say failed: {e}")
        return False

def synth_voice(text: str, workdir: Path) -> AudioFileClip:
    wav, aiff = workdir/"voice.wav", workdir/"voice.aiff"

    if USE_EDGE_TTS:
        # Try edge-tts first, then fall back to say
        if tts_edge_cli(text, wav):
            ac = AudioFileClip(str(wav))
            log(f"TTS(audio): source=edge-tts, dur={getattr(ac,'duration',None):.2f}s, fps={getattr(ac,'fps',None)}")
            return ac
        else:
            log("edge-tts failed or not available, falling back to mac 'say'...")

    # Either USE_EDGE_TTS is False, or edge-tts failed — use mac say
    if tts_mac_say(text, aiff):
        ac = AudioFileClip(str(aiff))
        log(f"TTS(audio): source=mac say, dur={getattr(ac,'duration',None):.2f}s, fps={getattr(ac,'fps',None)}")
        return ac

    raise RuntimeError("TTS failed: neither edge-tts nor macOS 'say' worked")

# ---------- Captions ----------
def parse_srt(srt_text: str):
    out = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        tl = next((ln for ln in lines if "-->" in ln), None)
        if not tl: continue
        m = re.findall(r"(\d+):(\d+):(\d+),(\d+)", tl)
        if len(m) < 2: continue
        def to_sec(t): h,mn,s,ms = map(int,t); return h*3600+mn*60+s+ms/1000.0
        t1,t2 = to_sec(m[0]), to_sec(m[1])
        t1 = max(0.0, min(t1, DURATION-0.05))
        t2 = max(t1+0.05, min(t2, DURATION-0.05))
        text = " ".join([ln for ln in lines if ln != tl][1:])
        out.append((t1,t2,text))
    log(f"SRT parsed: {len(out)} spans")
    return out

def auto_srt(text: str, total: float):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sents or len(" ".join(sents)) < 4: sents = [text]
    n = min(3,len(sents)); seg = total/n
    spans = [(i*seg, min(total-0.05,(i+1)*seg), sents[i]) for i in range(n)]
    log(f"Auto SRT: {len(spans)} spans")
    return spans

def has_imagemagick() -> bool:
    import shutil as _sh, os as _os
    present = bool(_os.getenv("IMAGEMAGICK_BINARY") or _sh.which("magick") or _sh.which("convert"))
    log(f"ImageMagick detected={present}, captions={'enabled' if ENABLE_CAPTIONS else 'disabled'}")
    return present

def render_captions_safe(base: VideoFileClip, spans):
    if not ENABLE_CAPTIONS or not has_imagemagick():
        return base
    overlays = []
    for (t1,t2,txt) in spans:
        if not txt.strip(): continue
        cap = (TextClip(txt, fontsize=48, color="white", font="Arial-Bold")
               .on_color(size=(FRAME_W-120,None), color=(0,0,0), col_opacity=0.4)
               .set_position(("center", FRAME_H-320))
               .set_start(t1).set_duration(max(0.1,t2-t1)))
        overlays.append(cap)
    out = CompositeVideoClip([base,*overlays]).set_fps(FPS)
    log(f"Captions overlay: {len(overlays)} clips, fps={getattr(out,'fps',None)}")
    return out

# ---------- Lower Third (no ImageMagick) ----------
def _safe_textlength(draw: ImageDraw.ImageDraw, s: str, font: ImageFont.ImageFont, fallback_px: int) -> float:
    try:
        L = draw.textlength(s, font=font)
        if L is None:
            return float(fallback_px)
        return float(L)
    except Exception:
        return float(fallback_px)

def _fit_text_lines(text, draw, font, max_width):
    avg_char_px = int((getattr(font, "size", 48) or 48) * 0.55)
    lines, line = [], ""
    for w in text.split():
        test = (line + " " + w).strip()
        w_px = _safe_textlength(draw, test, font, fallback_px=len(test) * avg_char_px)
        if w_px <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines[:3]

def make_lowerthird_overlay(body_text: str, duration: float) -> Optional[ImageClip]:
    try:
        W, H = FRAME_W, FRAME_H
        pad = 24
        box_w = int(W * 0.92)
        max_text_w = box_w - 2 * pad

        try:
            font = ImageFont.truetype("Arial.ttf", 48)
        except Exception:
            font = ImageFont.load_default()

        tmp = Image.new("RGBA", (box_w, 3000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tmp)
        lines = _fit_text_lines(body_text or "", draw, font, max_text_w)

        try:
            bbox = draw.textbbox((0, 0), "Ay", font=font)
            if bbox:
                line_h = (bbox[3] - bbox[1]) + 6
            else:
                line_h = int((getattr(font, "size", 48) or 48) * 1.2)
        except Exception:
            line_h = int((getattr(font, "size", 48) or 48) * 1.2)

        text_h = line_h * max(1, len(lines))
        box_h = text_h + 2 * pad

        img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 120))
        d = ImageDraw.Draw(img)
        y = pad
        for ln in lines:
            d.text((pad, y), ln, font=font, fill=(255, 255, 255, 255))
            y += line_h

        x = (W - box_w) // 2
        y = H - box_h - 140

        arr = np.array(img)
        clip = ImageClip(arr).set_position((int(x), int(y))).set_duration(float(duration)).set_fps(FPS)
        log(f"Lower-third: lines={len(lines)}, box={box_w}x{box_h}, fps={getattr(clip,'fps',None)}")
        return clip
    except Exception as e:
        print(f"[WARN] lower-third disabled: {e}")
        return None

# ---------- Telegram ----------
def tg_send_video(bot_token, chat_id, mp4_path, caption=""):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
        with mp4_path.open("rb") as f:
            files = {"video": (mp4_path.name, f, "video/mp4")}
            data = {"chat_id": chat_id, "caption": caption}
            r = requests.post(url, data=data, files=files, timeout=120)
        ok = (r.status_code == 200)
        log(f"Telegram send to {chat_id}: status={r.status_code}, ok={ok}")
        return ok
    except Exception as e:
        log(f"Telegram send error: {e}")
        return False

# ---------- Main pipeline ----------
def build_video(voice_text, maybe_srt, title_hint) -> Path:
    print("\n========== build_video START ==========")
    print(f"voice_text[0:80]={repr((voice_text or '')[:80])}")
    print(f"title_hint={repr(title_hint)} | DURATION={DURATION} | FPS={FPS} | FRAME={FRAME_W}x{FRAME_H}")

    work = Path(tempfile.mkdtemp(prefix="work_"))
    print(f"[build] workdir={work}")

    # 1) TTS
    print("[build] TTS: synthesizing...")
    voice = synth_voice(voice_text, work)
    print(f"[build] TTS done: type={type(voice)}, dur={getattr(voice,'duration',None):.3f}s")

    a_dur = float(voice.duration or 0.0)
    target = DURATION
    EPS = 0.05
    if a_dur > target:
        print(f"[build] TTS longer than target ({a_dur:.3f} > {target}); trimming to {target - EPS:.3f}s")
        voice = voice.subclip(0, target - EPS)
        a_dur = target - EPS
    print(f"[build] TTS final dur={a_dur:.3f}s")

    # 2) B-roll
    query = keyword_guess(title_hint or voice_text)
    print(f"[build] B-roll: query='{query}' for duration={target}")
    base_bg = assemble_broll(query, target)
    print(f"[build] B-roll ready: size={base_bg.w}x{base_bg.h}, dur={base_bg.duration:.1f}, fps={getattr(base_bg,'fps',None)}")

    # 3) Audio padding/clamp
    print("[build] Audio: padding/clamp...")
    if a_dur < target - EPS:
        pad = (target - a_dur)
        print(f"[build] Adding silence: {pad:.3f}s")
        silence = AudioClip(lambda t: 0, duration=pad).set_fps(44100)
        audio = CompositeAudioClip([voice, silence.set_start(a_dur)]).set_duration(target - EPS)
    else:
        print(f"[build] No silence needed; setting voice duration={target - EPS:.3f}s")
        audio = voice.set_duration(target - EPS)
    print(f"[build] Audio track done: dur={getattr(audio,'duration',None)}")

    # 4) Base composite
    print("[build] Attaching audio to base background...")
    base = base_bg.set_audio(audio).set_duration(target)
    print(f"[build] Base composite: size={base.w}x{base.h}, dur={base.duration:.1f}, fps={getattr(base,'fps',None)}")

    # 5) Lower-third overlay (first sentence) — fully optional
    first_sentence = re.split(r"(?<=[.!?])\s+", (voice_text or "").strip())[0][:220] if (voice_text or "").strip() else ""
    print(f"[build] Lower-third: first_sentence[0:120]={repr(first_sentence[:120])}")
    lt = make_lowerthird_overlay(first_sentence, target)
    if lt is not None:
        print(f"[build] Lower-third clip: size={lt.w}x{lt.h}, dur={lt.duration}, fps={getattr(lt,'fps',None)}")
        base = CompositeVideoClip([base, lt]).set_fps(FPS)
        print(f"[build] After LT composite: size={base.w}x{base.h}, dur={base.duration}, fps={getattr(base,'fps',None)}")
    else:
        print("[build] Lower-third skipped.")

    # 6) Captions (optional)
    print(f"[build] Captions: ENABLE_CAPTIONS={ENABLE_CAPTIONS}")
    spans = parse_srt(maybe_srt) if (maybe_srt and maybe_srt.strip()) else auto_srt(voice_text, target)
    final = render_captions_safe(base, spans).set_fps(FPS)
    print(f"[build] Final before export: size={final.w}x{final.h}, dur={final.duration:.1f}, fps={getattr(final,'fps',None)}")

    # 7) Export
    out_name = slug(title_hint or voice_text[:40]) or f"short_{int(time.time())}"
    out_path = OUT_DIR / f"{out_name}.mp4"
    print(f"[build] Writing file: {out_path.name}")
    final.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=FPS,
        bitrate="4500k",
        preset="medium",
        threads=4,
        temp_audiofile=str(work / 'temp_aac.m4a'),
        remove_temp=True,
        verbose=False,
        logger=None
    )
    print(f"[build] Wrote: {out_path.name} ({out_path.stat().st_size} bytes)")

    # Cleanup
    try:
        base_bg.close(); final.close(); voice.close()
    except Exception as e:
        log(f"cleanup note: {e}")
    shutil.rmtree(work, ignore_errors=True)
    return out_path

def process_one(txt_path, srt_path):
    try:
        text = txt_path.read_text().strip()
        print(f"got text: {text}")
        srt_text = srt_path.read_text() if srt_path and srt_path.exists() else None
        print(f"got srt text: {srt_text}")
        title_hint = txt_path.stem.replace("_", " ")
        print(f"got title hint: {title_hint}")
        log(f"PROCESS: {txt_path.name} (has_srt={'yes' if srt_text else 'no'})")
        return build_video(text, srt_text, title_hint)
    except Exception as ex:
        print(f"[ERROR] {txt_path.name}: {ex}")
        return None

def move_done(txt, srt, mp4):
    try:
        DONE_DIR.mkdir(exist_ok=True)
        shutil.move(str(txt), DONE_DIR / txt.name)
        if srt and srt.exists(): shutil.move(str(srt), DONE_DIR / srt.name)
        if mp4 and mp4.exists(): shutil.copy2(str(mp4), DONE_DIR / mp4.name)
    except Exception as e:
        log(f"move_done note: {e}")

def move_failed(txt, srt):
    try:
        FAILED_DIR.mkdir(exist_ok=True)
        shutil.move(str(txt), FAILED_DIR / txt.name)
        if srt and srt.exists(): shutil.move(str(srt), FAILED_DIR / srt.name)
    except Exception as e:
        log(f"move_failed note: {e}")

def main():
    scripts = find_scripts()
    if not scripts:
        print(f"No scripts found in {IN_DIR}/")
        return
    targets = read_targets() if SEND_TELEGRAM and TELEGRAM_BOT_TOKEN else []
    print(f"Found {len(scripts)} script(s). Telegram: {'on' if targets else 'off'}")
    for txt in tqdm(scripts, desc="Rendering"):
        txt, srt = stem_pair(txt)
        mp4 = process_one(txt, srt)
        if mp4:
            sent_all = True
            if targets:
                cap = txt.stem.replace("_", " ")
                for chat in targets:
                    ok = tg_send_video(TELEGRAM_BOT_TOKEN, chat, mp4, caption=cap)
                    sent_all = sent_all and ok
                    time.sleep(SEND_COOLDOWN_SEC)
            move_done(txt, srt, mp4)
            if targets and not sent_all:
                print(f"[WARN] Telegram send failed for {mp4.name}")
        else:
            move_failed(txt, srt)

if __name__ == "__main__":
    main()
