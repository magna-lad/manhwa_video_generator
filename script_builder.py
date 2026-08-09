""" 
To make a meaningful script with the cluster of JSON and character registry
"""

import ollama
import json
import os
import re
from tqdm import tqdm

# CONFIG

TEXT_MODEL = "llama3.2-vision:11b"  # swap for whichever text-only model you have pulled (e.g. "llama3.2", "qwen3")
SCRIPT_PROGRESS_DIR_NAME = r""

CHAPTER_SCRIPT_OUTPUT     = "chapter_script.json"
VOICE_MAP_FILENAME        = "character_voice_map.json"

NARRATOR_NAME  = "Narrator"
NARRATOR_VOICE = "am_adam"

# Confirmed Kokoro voice IDs — check hexgrad/Kokoro-82M/VOICES.md on HuggingFace
# for the full list and to swap in ones that fit your characters better.
VOICE_POOL = ["af_heart", "af_bella", "af_sarah", "am_michael", "bf_emma"]

# Emotion -> deterministic TTS delivery params.
# Kokoro has no emotion/SSML tags — it only reads voice + speed + punctuation for prosody.
# So "emotion" here is OUR metadata, used to pick speed/pauses/voice — never sent to Kokoro as a tag.
EMOTION_TTS_PARAMS = {
    "neutral":    {"speed": 1.00, "pause_after_ms": 300},
    "happy":      {"speed": 1.05, "pause_after_ms": 250},
    "sad":        {"speed": 0.85, "pause_after_ms": 500},
    "angry":      {"speed": 1.15, "pause_after_ms": 200},
    "scared":     {"speed": 1.10, "pause_after_ms": 200},
    "shouting":   {"speed": 1.10, "pause_after_ms": 250},
    "whispering": {"speed": 0.80, "pause_after_ms": 400},
    "ominous":    {"speed": 0.75, "pause_after_ms": 600},
    "comedic":    {"speed": 1.05, "pause_after_ms": 200},
    "tense":      {"speed": 0.95, "pause_after_ms": 350},
    "surprised":  {"speed": 1.10, "pause_after_ms": 250},
    "sarcastic":  {"speed": 0.95, "pause_after_ms": 350},
}
EMOTION_ENUM = list(EMOTION_TTS_PARAMS.keys())

WORDS_PER_SECOND = 3

# VOICE MAP — persistent, one voice per character for the whole chapter/story

def load_voice_map(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_voice_map(path, voice_map):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(voice_map, f, indent=2, ensure_ascii=False)


def get_or_assign_voice(character_name, voice_map):
    if character_name == NARRATOR_NAME:
        return NARRATOR_VOICE
    if character_name in voice_map:
        return voice_map[character_name]
    used = set(voice_map.values())
    for v in VOICE_POOL:
        if v not in used:
            voice_map[character_name] = v
            return v
    # pool exhausted — cycle through it rather than crash
    voice_map[character_name] = VOICE_POOL[len(voice_map) % len(VOICE_POOL)]
    return voice_map[character_name]



# LLM SCRIPT GENERATION (per page)

def build_prompt(page_data, known_characters):
    known_list = ", ".join(known_characters) if known_characters else "(none yet)"
    return f"""
You are a Script Adaptation Agent for an AI-narrated manhwa recap. You take structured
page analysis (JSON) and turn it into a clean narration/dialogue script ready for text-to-speech.

KNOWN CHARACTERS SO FAR: {known_list}

RULES:
- For each entry in transcribed_text, produce one script line.
- speaker: match speaker_description to one of the KNOWN CHARACTERS if it clearly refers to
  the same person; otherwise keep the original speaker_description as-is (a new character).
- Narration boxes (type=narration) get speaker="{NARRATOR_NAME}".
- Clean up the text slightly for spoken delivery (fix OCR artifacts, keep meaning and wording
  as close to the original as possible). Do NOT invent new dialogue.
- emotion: pick exactly one value from this list based on mood_and_atmosphere, narrator_hints,
  and the line's own content: {EMOTION_ENUM}
- include_in_audio: true for speech/thought/narration. false for sound_effect and floating_text
  (these are visual-only and should not be voiced by TTS).
- Return ONLY a JSON object, no extra text, no markdown fences.

REQUIRED JSON SCHEMA:
{{
  "lines": [
    {{
      "speaker": "string",
      "type": "speech | thought | narration | sound_effect | floating_text",
      "text": "cleaned spoken-ready text",
      "emotion": "one of {EMOTION_ENUM}",
      "include_in_audio": true
    }}
  ]
}}

PAGE DATA:
{json.dumps(page_data, ensure_ascii=False)}
"""


def generate_page_script(page_data, known_characters, model=TEXT_MODEL):
    prompt = build_prompt(page_data, known_characters)
    response = ollama.chat(
        model=model,
        format="json",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    raw = response["message"]["content"]
    cleaned = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        tqdm.write(f"[ERROR] Script JSON parse failed: {e}")
        tqdm.write(f"Raw:\n{raw}\n")
        return None


def enrich_line(line, voice_map):
    """Attach deterministic TTS params (voice, speed, pause, duration estimate) to an LLM-produced line."""
    speaker = line.get("speaker", "Unknown")
    emotion = line.get("emotion", "neutral")
    if emotion not in EMOTION_TTS_PARAMS:
        emotion = "neutral"
    params = EMOTION_TTS_PARAMS[emotion]
    voice = get_or_assign_voice(speaker, voice_map)

    word_count = len(line.get("text", "").split())
    est_duration = round((word_count / WORDS_PER_SECOND) / params["speed"], 2) if word_count else 0.0

    line["voice_id"] = voice
    line["speed"] = params["speed"]
    line["pause_after_ms"] = params["pause_after_ms"]
    line["estimated_duration_sec"] = est_duration
    return line


# PER-PAGE INCREMENTAL SAVE (same crash-safety pattern as your OCR script)


def get_script_progress_path(chapter_folder, page_index):
    d = os.path.join(chapter_folder, SCRIPT_PROGRESS_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"page_{page_index:03d}.json")


def save_page_script(chapter_folder, page_index, data):
    with open(get_script_progress_path(chapter_folder, page_index), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_page_script(chapter_folder, page_index):
    path = get_script_progress_path(chapter_folder, page_index)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# PROCESS ONE CHAPTER

def process_chapter_script(chapter_folder):
    metadata_path = os.path.join(chapter_folder, r"")
    if not os.path.exists(metadata_path):
        print(f"[SKIP] No chapter_metadata.json in {chapter_folder}")
        return []

    with open(metadata_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    voice_map_path = os.path.join(chapter_folder, VOICE_MAP_FILENAME)
    voice_map = load_voice_map(voice_map_path)
    known_characters = set(voice_map.keys())

    chapter_script = []
    pbar = tqdm(enumerate(pages, start=1), total=len(pages), desc="Scripting", unit="page")

    for page_index, page_data in pbar:
        existing = load_page_script(chapter_folder, page_index)
        if existing:
            chapter_script.append(existing)
            pbar.set_postfix(page=page_index, status="cached")
            continue

        pbar.set_postfix(page=page_index, status="processing")
        result = generate_page_script(page_data, sorted(known_characters))

        if not result or "lines" not in result:
            tqdm.write(f"[WARN] Skipping page {page_index} — script generation failed.")
            continue

        enriched_lines = [enrich_line(line, voice_map) for line in result["lines"]]
        known_characters.update(l["speaker"] for l in enriched_lines if l["speaker"] != NARRATOR_NAME)

        page_script = {
            "page_number": page_index,
            "page_type": page_data.get("page_type", "standard"),
            "panel_count": page_data.get("panel_count"),
            "overall_mood": page_data.get("mood_and_atmosphere", ""),
            "lines": enriched_lines,
            "estimated_duration_sec": round(
                sum(l["estimated_duration_sec"] + l["pause_after_ms"] / 1000 for l in enriched_lines), 2
            ),
        }

        save_page_script(chapter_folder, page_index, page_script)
        save_voice_map(voice_map_path, voice_map)  # persist as we go — same crash-safety spirit
        chapter_script.append(page_script)

    output_path = os.path.join(chapter_folder, CHAPTER_SCRIPT_OUTPUT)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chapter_script, f, indent=2, ensure_ascii=False)
    tqdm.write(f"[SAVED] {output_path} ({len(chapter_script)} pages)")

    return chapter_script


if __name__ == "__main__":
    chapter_folder = r""
    process_chapter_script(chapter_folder)