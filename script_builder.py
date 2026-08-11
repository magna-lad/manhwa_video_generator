""" 
To make a meaningful script with the cluster of JSON and character registry.
Includes strict anti-hallucination validation and automatic retry logic.
"""

import ollama
import json
import os
import re
import difflib
from tqdm import tqdm
import random

# CONFIG

TEXT_MODEL = "llama3.2-vision:11b"  # swap for whichever text-only model you have pulled (e.g. "llama3.2", "qwen3")
MAX_RETRIES = 3                     # Number of times to re-prompt the LLM if it hallucinates or fails validation

# Paths (Update if necessary)
SCRIPT_PROGRESS_DIR_NAME = r""
CHAPTER_SCRIPT_OUTPUT    = "chapter_script.json"
VOICE_MAP_FILENAME       = "character_voice_map.json"

NARRATOR_NAME  = "Narrator"
NARRATOR_VOICE = "am_adam"

# Confirmed Kokoro voice IDs
VOICE_POOL = ["af_heart", "af_bella", "af_sarah", "am_michael", "bf_emma"]

# Emotion -> deterministic TTS delivery params.
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
WORDS_PER_SECOND = 4


# VOICE MAP

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


# LLM SCRIPT GENERATION

def build_prompt(page_data, known_characters, previous_lines_context=None):
    known_list = ", ".join(known_characters) if known_characters else "(none yet)"
    
    # SANITIZE INPUT: Isolate only the transcribed text. Do NOT pass the bloated 
    # visual_description which might contain repetition loops that confuse the LLM.
    source_lines = page_data.get("transcribed_text", [])
    
    # Assign an ID to every line to force a strict 1-to-1 mechanical map in the output
    clean_input_lines = []
    for i, line in enumerate(source_lines):
        clean_input_lines.append({
            "line_id": i,
            "type": line.get("type", "unknown"),
            "speaker_hint": line.get("speaker_description", "None"),
            "raw_text": line.get("text", "")
        })

    # Tell the LLM what just happened on the last page to keep context flowing (optional)
    context_str = ""
    if previous_lines_context:
        context_str = f"PREVIOUS PAGE CONTEXT (Use to identify speakers, but do NOT include in output):\n{json.dumps(previous_lines_context, ensure_ascii=False)}\n\n"

    return f"""You are a strict Data Transformation Agent for an AI-narrated manhwa recap. 
Your ONLY job is to clean up raw OCR text and format it into text-to-speech JSON.

KNOWN CHARACTERS: {known_list}
PERMITTED EMOTIONS: {EMOTION_ENUM}

{context_str}INPUT DATA TO TRANSFORM:
{json.dumps(clean_input_lines, ensure_ascii=False, indent=2)}

CRITICAL RULES - READ CAREFULLY OR YOU WILL FAIL:
1. NO HALLUCINATIONS: You must process EVERY item in the INPUT DATA. Do NOT add new lines. Do NOT skip lines. Do NOT invent dialogue.
2. STRICT MAPPING: The output "lines" array must have EXACTLY {len(clean_input_lines)} items.
3. TEXT FIDELITY & TTS DIRECTING: Keep the core wording identical, BUT you MUST improve the punctuation for the text-to-speech engine. 
   - Use ellipses (...) for dramatic pauses or trailing off.
   - Use dashes (-) for abrupt stops or quick shifts.
   - Use commas (,) to slow the pacing down naturally inside sentences.
   - Use exclamation marks (!) for shouting/excitement.
4. ID MATCHING: You MUST include "line_id" in each output object, matching the corresponding input item.

OUTPUT SCHEMA (Return ONLY valid JSON):
{{
  "lines": [
    {{
      "line_id": int (must match input),
      "speaker": "string (Use Narrator for narration, or infer from KNOWN CHARACTERS / speaker_hint)",
      "type": "speech | thought | narration | sound_effect | floating_text",
      "text": "cleaned text",
      "emotion": "string (MUST be one of the PERMITTED EMOTIONS)",
      "include_in_audio": boolean (true for speech/thought/narration, false for sound_effect/floating_text)
    }}
  ]
}}
"""

def text_similarity(str1, str2):
    """Returns a float 0.0 to 1.0 representing how similar two strings are."""
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def generate_page_script(page_data, known_characters, previous_lines_context=None, model=TEXT_MODEL):
    source_lines = page_data.get("transcribed_text", [])
    if not source_lines:
        return {"lines": []}  # Quick exit if the page has no dialogue/text

    prompt = build_prompt(page_data, known_characters, previous_lines_context)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.chat(
                model=model,
                format="json",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2}, # Low temp for data formatting, but non-zero to prevent getting stuck
            )
            
            raw = response["message"]["content"]
            cleaned = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            result = json.loads(cleaned)

            # PYTHON SAFETY NET VALIDATION
            
            if "lines" not in result:
                raise ValueError("JSON is missing the required 'lines' array.")

            output_lines = result["lines"]
            
            # 1. Check Cardinality (Count mismatch)
            if len(output_lines) != len(source_lines):
                raise ValueError(f"Cardinality mismatch: Expected {len(source_lines)} lines, got {len(output_lines)}")

            # 2. Check for Hallucinations and ID alignment
            for idx, (source, output) in enumerate(zip(source_lines, output_lines)):
                # Patch missing or incorrect IDs to ensure zip alignment is correct
                if output.get("line_id", -1) != idx:
                    output["line_id"] = idx 
                
                out_text = output.get("text", "")
                src_text = source.get("text", "")
                
                # If the output text radically changed from the source text, it's a hallucination
                if len(src_text) > 3 and len(out_text) > 3: 
                    similarity = text_similarity(src_text, out_text)
                    if similarity < 0.25:  # Less than 25% similar
                        raise ValueError(f"Hallucination detected. Source: '{src_text}', Output: '{out_text}'")

                # Validate Emotion constraints
                if output.get("emotion") not in EMOTION_ENUM:
                    output["emotion"] = "neutral"

            # If all checks pass, break out of retry loop
            return result

        except json.JSONDecodeError as e:
            tqdm.write(f"\n[WARN] Attempt {attempt+1} - JSON parse failed: {e}")
        except ValueError as e:
            tqdm.write(f"\n[WARN] Attempt {attempt+1} - Validation failed: {e}")
        
    tqdm.write(f"[ERROR] Failed to generate a valid script for this page after {MAX_RETRIES} attempts.")
    return None


def enrich_line(line, voice_map):
    """Attach deterministic TTS params (voice, speed, pause, duration estimate) to an LLM-produced line."""
    speaker = line.get("speaker", "Unknown")
    emotion = line.get("emotion", "neutral")
    if emotion not in EMOTION_TTS_PARAMS:
        emotion = "neutral"
    params = EMOTION_TTS_PARAMS[emotion]
    voice = get_or_assign_voice(speaker, voice_map)

    jitter = random.uniform(0.98, 1.02)
    final_speed = round(params["speed"] * jitter, 3)

    word_count = len(line.get("text", "").split())
    est_duration = round((word_count / WORDS_PER_SECOND) / final_speed, 2) if word_count else 0.0

    line["voice_id"] = voice
    line["speed"] = params["speed"]
    line["pause_after_ms"] = params["pause_after_ms"]
    line["estimated_duration_sec"] = est_duration
    
    # We can remove the line_id here as we don't need it in the final JSON
    line.pop("line_id", None) 
    
    return line


# FILE HANDLING

def get_script_progress_path(chapter_folder, page_index):
    d = os.path.join(chapter_folder, os.path.basename(SCRIPT_PROGRESS_DIR_NAME)) # Just use the basename for cross-compatibility
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


# MAIN LOOP

def process_chapter_script(chapter_folder):
    # Fixed Path reference: It now looks inside whatever folder you pass into the function, instead of being hardcoded.
    metadata_path = os.path.join(chapter_folder, "chapter_metadata.json")
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
        
        # Pull context from the previous page to prevent disjointed conversation tracking
        prev_context = None
        if chapter_script and chapter_script[-1].get("lines"):
            # Get the last 3 spoken lines for context
            prev_context = [{"speaker": l["speaker"], "text": l["text"]} for l in chapter_script[-1]["lines"][-3:]]

        # Trigger the LLM Request
        result = generate_page_script(page_data, sorted(known_characters), previous_lines_context=prev_context)

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
        save_voice_map(voice_map_path, voice_map)  # persist as we go
        chapter_script.append(page_script)

    output_path = os.path.join(chapter_folder, CHAPTER_SCRIPT_OUTPUT)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chapter_script, f, indent=2, ensure_ascii=False)
    tqdm.write(f"[SAVED] {output_path} ({len(chapter_script)} pages)")

    return chapter_script


if __name__ == "__main__":
    target_chapter_folder = r""
    process_chapter_script(target_chapter_folder)