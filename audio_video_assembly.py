import os
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm

from kokoro import KPipeline
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# CONFIG

KOKORO_LANG_CODE = "a"   # 'a' = American English — matches the af_/am_ voice IDs from script_generator.py
SAMPLE_RATE      = 24000  # Kokoro's native output rate

AUDIO_PROGRESS_DIR_NAME = "_audio_progress"   # per-page rendered .wav files (crash-safe, like your other stages)
FINAL_VIDEO_NAME        = "chapter_video.mp4"
FPS                      = 24

CHAPTER_SCRIPT_FILENAME = "chapter_script.json"


# KOKORO SETUP (load once, reuse across all lines/pages)

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)
    return _pipeline


def _to_numpy(audio):
    """Kokoro can yield torch tensors depending on backend — normalize to float32 numpy."""
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def synthesize_line(text, voice_id, speed):
    """Returns a 1D float32 numpy array, or None if Kokoro produced nothing for this text."""
    pipeline = get_pipeline()
    chunks = []
    for _, _, audio in pipeline(text, voice=voice_id, speed=speed):
        if audio is None:
            continue
        chunks.append(_to_numpy(audio))
    if not chunks:
        return None
    return np.concatenate(chunks)


def build_silence(pause_ms):
    n_samples = int(SAMPLE_RATE * pause_ms / 1000)
    return np.zeros(n_samples, dtype=np.float32)


# PER-PAGE AUDIO — crash-safe, mirrors your OCR/script stages

def get_audio_progress_path(chapter_folder, page_number):
    d = os.path.join(chapter_folder, AUDIO_PROGRESS_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"page_{page_number:03d}.wav")


def synthesize_page_audio(page_script):
    """Concatenates every voiced line's audio + its pause into one page-length waveform."""
    segments = []
    for line in page_script.get("lines", []):
        if not line.get("include_in_audio", True):
            continue
        text = (line.get("text") or "").strip()
        if not text:
            continue

        audio = synthesize_line(text, line["voice_id"], line["speed"])
        if audio is None or len(audio) == 0:
            tqdm.write(f"[WARN] No audio produced for line: {text[:60]!r}")
            continue

        segments.append(audio)
        segments.append(build_silence(line.get("pause_after_ms", 300)))

    if not segments:
        return None
    return np.concatenate(segments)


def process_chapter_audio(chapter_folder, chapter_script):
    pbar = tqdm(chapter_script, desc="Synthesizing audio", unit="page")
    for page_script in pbar:
        page_number = page_script["page_number"]
        out_path = get_audio_progress_path(chapter_folder, page_number)

        if os.path.exists(out_path):
            pbar.set_postfix(page=page_number, status="cached")
            continue

        pbar.set_postfix(page=page_number, status="rendering")
        audio = synthesize_page_audio(page_script)
        if audio is None:
            tqdm.write(f"[WARN] Page {page_number} produced no audio at all — skipping.")
            continue

        sf.write(out_path, audio, SAMPLE_RATE)


# VIDEO ASSEMBLY

def get_sorted_images(chapter_folder):
    return sorted(
        f for f in os.listdir(chapter_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(chapter_folder, f))
    )


def build_chapter_video(chapter_folder, chapter_script):
    images = get_sorted_images(chapter_folder)

    if len(images) != len(chapter_script):
        tqdm.write(
            f"[WARN] {len(images)} images vs {len(chapter_script)} script pages — "
            f"they should match 1:1 in page order. Proceeding on the shorter of the two."
        )

    clips = []
    for page_script, img_name in zip(chapter_script, images):
        page_number = page_script["page_number"]
        audio_path = get_audio_progress_path(chapter_folder, page_number)

        if not os.path.exists(audio_path):
            tqdm.write(f"[WARN] No rendered audio for page {page_number} — skipping this page in the video.")
            continue

        audio_clip = AudioFileClip(audio_path)
        img_path = os.path.join(chapter_folder, img_name)

        # Duration comes from the ACTUAL rendered audio, not the estimate in chapter_script.json —
        # real TTS output length drifts slightly from the word-count estimate, and this keeps image/audio in sync.
        image_clip = ImageClip(img_path).set_duration(audio_clip.duration).set_audio(audio_clip)
        clips.append(image_clip)

    if not clips:
        raise RuntimeError("No page clips were built — check that audio rendering succeeded first.")

    final = concatenate_videoclips(clips, method="compose")
    output_path = os.path.join(chapter_folder, FINAL_VIDEO_NAME)
    final.write_videofile(output_path, fps=FPS, codec="libx264", audio_codec="aac")

    for c in clips:
        c.close()
    final.close()

    return output_path


# MAIN

def process_chapter_full(chapter_folder):
    script_path = os.path.join(chapter_folder, CHAPTER_SCRIPT_FILENAME)
    if not os.path.exists(script_path):
        print(f"[SKIP] No {CHAPTER_SCRIPT_FILENAME} in {chapter_folder}")
        return

    with open(script_path, "r", encoding="utf-8") as f:
        chapter_script = json.load(f)

    process_chapter_audio(chapter_folder, chapter_script)
    output_path = build_chapter_video(chapter_folder, chapter_script)
    print(f"[DONE] Video saved to {output_path}")


if __name__ == "__main__":
    chapter_folder = r""
    process_chapter_full(chapter_folder)