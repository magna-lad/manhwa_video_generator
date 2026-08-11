import ollama
import base64
import json
import os
import re
from tqdm import tqdm


import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# CONFIG

PROGRESS_DIR_NAME = "_ocr_progress"   # subfolder inside each chapter folder
MASTER_OUTPUT_FILE = "all_chapters_merged.json"  # final merged output (saved in parent dir)
REGISTRY_FILENAME   = "character_registry.json"   #to maintain registry of a character -> to give them a particular voice



def image_ocr(path):
    
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        response = ollama.chat(
            model='llama3.2-vision:11b',
            format='json',
            messages=[{
                'role': 'user',
                'content': 
                            """
                            You are a Manhwa/Webtoon Analyst and Transcriber. Your job is to analyze panels from a manhwa chapter and extract maximum textual and visual information.

                            VERY IMPORTANT: Return ONLY the JSON object. No preamble, no markdown fences, no commentary. Keys must be exactly: page_type, mood_and_atmosphere, visual_description, transcribed_text, narrator_hints.
                            Do not hallucinate text or events that are not visibly present in the image.

                            READING ORDER (transcribed_text):
                            Modern Korean manhwa/webtoons are formatted for LEFT-TO-RIGHT, top-to-bottom reading — the opposite of traditional right-to-left manga. Default to this order:
                              1. Top of page to bottom.
                              2. Left to right within the same horizontal band.
                            If bubble tails, gutter flow, or panel layout suggest the page actually follows right-to-left convention instead (some scanlations or traditional-style manhwa do), transcribe in that order instead and say so explicitly in narrator_hints — do not silently switch conventions.
                            Number each bubble sequentially in the order you transcribe it. If bubble order is genuinely ambiguous (e.g., two bubbles stacked with no clear vertical offset), note the ambiguity in narrator_hints rather than guessing silently.

                            CHARACTER CONSISTENCY:
                            When a speaker is not named on the page, describe them with a short, distinctive tag (e.g., "Red-Haired Man", "Scarred Woman"). Reuse the exact same tag for the same character every time they appear on this page — don't vary the phrasing mid-page, since downstream narration keys off this string for voice continuity.

                            SOUND EFFECTS / STYLIZED TEXT:
                            For SFX (e.g. 쿵, 휙), transcribe the original text and give a short bracketed sense-gloss, e.g. "쿵 [heavy thud]". Never invent an SFX that isn't visibly drawn on the page.

                            Adapt extraction depth based on page_type:

                            1. "standard" (dialogue-driven pages):
                               - Transcribe every speech bubble, narration box, and thought bubble.
                               - Identify the speaker (name if known, else a consistent descriptive tag).
                               - Describe character actions, facial expressions, and spatial positioning as they relate to who's speaking.

                            2. "images_only" (action/scenery, no legible text):
                               - transcribed_text must be an empty array.
                               - Focus on choreography, kinetic energy, weapon/impact points, camera angle (e.g., "low angle looking up at the monster"), and emotional beats conveyed purely visually.

                            3. "text_only" (blank/minimal background, text-forward):
                               - Transcribe with 100% accuracy — this is the highest-stakes field on this page type.
                               - Note background color/texture and text styling (e.g., "white text on pitch-black background, jagged font indicating a scream").

                            4. "artistic_spread" (full-page or multi-panel epic reveal, wide vertical art):
                               - Read strictly top to bottom.
                               - Emphasize scale, lighting, color palette, and the dramatic/awe intent of the composition. Note any text but keep the visual description primary.

                            5. "photo_insert" (a real photograph or non-illustrated reference image embedded in the chapter, e.g. author's note, real-world reference photo):
                               - Describe the photo's literal content plainly; do not treat it as part of the story's diegetic art unless context clearly indicates otherwise.
                               - transcribed_text should capture any caption or overlaid text only.

                            REQUIRED JSON SCHEMA (always return all five keys; transcribed_text is [] when there is no text on the page):

                            {
                              "page_type": "standard | images_only | text_only | artistic_spread | photo_insert",
                              "mood_and_atmosphere": "Brief description of lighting, color palette, and emotional tone",
                              "visual_description": "Detailed, chronological (top-to-bottom) description of what is happening visually",
                              "transcribed_text": [
                                {
                                  "type": "speech | thought | narration | sound_effect | floating_text",
                                  "speaker_description": "Who is saying this, or 'None' for narration/SFX",
                                  "text": "The exact transcribed text, original + gloss for SFX"
                                }
                              ],
                              "narrator_hints": "Suggestions for the narrator AI (e.g. 'long pause here', 'fast-paced action scene', 'ominous reveal', or a note about reading-order ambiguity)"
                            }
                        }""",
                'images': [img_b64]
            }],
             options={
                'temperature': 0.1           # lower = more deterministic JSON
            }
        )
        raw=response["message"]["content"]

        # 2. Safety Net: Strip markdown formatting if the model hallucinates it (```json ... ```)
        cleaned_raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
        cleaned_raw = re.sub(r"\s*```$", "", cleaned_raw).strip()
        try:
            data = json.loads(cleaned_raw)
            return data
            #print(json.dumps(data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            tqdm.write(f"\n[ERROR] Failed to parse JSON for {os.path.basename(path)}. Error: {e}")
            tqdm.write(f"Raw Output:\n{raw}\n")
            #print(raw)



def get_file_paths(folder_path):
    json_per_page = []
    
    files= sorted(os.listdir(folder_path))
    pbar = tqdm(files, desc="Processing Chapter", unit="page")

    for name in pbar:
        full_path = os.path.join(folder_path, name)
        if full_path.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(full_path):
            pbar.set_postfix(file=name) 
            result=image_ocr(full_path)
            if result:
                json_per_page.append(result)
    
    return json_per_page



# PER-PAGE INCREMENTAL SAVE

def get_progress_path(chapter_folder, filename):
    """Returns the path where a single page's JSON is saved."""
    progress_dir = os.path.join(chapter_folder, PROGRESS_DIR_NAME)
    os.makedirs(progress_dir, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    return os.path.join(progress_dir, f"{stem}.json")


def save_page_json(chapter_folder, filename, data):
    path = get_progress_path(chapter_folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_page_json(chapter_folder, filename):
    path = get_progress_path(chapter_folder, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None





# PROCESS ONE CHAPTER

def process_chapter(chapter_folder):
    """
    Process all images in a chapter folder.
    - Skips pages already saved (crash recovery).
    - Saves each page immediately after OCR.
    - Merges all pages into chapter_metadata.json at the end.
    """
    files = sorted([
        f for f in os.listdir(chapter_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(chapter_folder, f))
    ])

    if not files:
        print(f"[SKIP] No images found in {chapter_folder}")
        return []

    chapter_name = os.path.basename(chapter_folder)
    json_per_page = []
    skipped = 0

    pbar = tqdm(files, desc=f"Chapter: {chapter_name}", unit="page")

    for name in pbar:
        full_path = os.path.join(chapter_folder, name)

        # ── Crash recovery: load existing result if already done ──
        existing = load_page_json(chapter_folder, name)
        if existing:
            json_per_page.append(existing)
            skipped += 1
            pbar.set_postfix(file=name, status="cached")
            continue

        # ── Run OCR ──
        pbar.set_postfix(file=name, status="processing")
        result = image_ocr(full_path)

        if result:
            # ── Save immediately (crash safety) ──
            save_page_json(chapter_folder, name, result)
            json_per_page.append(result)
        else:
            tqdm.write(f"[WARN] Skipping {name} due to OCR failure.")

    if skipped:
        tqdm.write(f"[INFO] {skipped}/{len(files)} pages loaded from cache.")

    # ── Merge into chapter_metadata.json ──
    chapter_output = os.path.join(chapter_folder, "chapter_metadata.json")
    with open(chapter_output, "w", encoding="utf-8") as f:
        json.dump(json_per_page, f, indent=2, ensure_ascii=False)
    tqdm.write(f"[SAVED] {chapter_name} → chapter_metadata.json ({len(json_per_page)} pages)")

    return json_per_page



# PROCESS MULTIPLE CHAPTERS + FINAL MERGE

def process_all_chapters(chapter_folders, master_output_dir=None):
    """
    Process a list of chapter folders and merge all into one master JSON.
    master_output_dir defaults to the parent of the first chapter folder.
    """
    all_chapters = {}

    for folder in chapter_folders:
        folder = os.path.normpath(folder)
        chapter_name = os.path.basename(folder)
        print(f"\n{'='*50}")
        print(f"Processing: {chapter_name}")
        print(f"{'='*50}")

        pages = process_chapter(folder)
        all_chapters[chapter_name] = pages

    # ── Save master merged JSON ──
    if master_output_dir is None:
        master_output_dir = os.path.dirname(chapter_folders[0])

    master_path = os.path.join(master_output_dir, MASTER_OUTPUT_FILE)
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(all_chapters, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"[DONE] All chapters merged → {master_path}")
    print(f"Total chapters: {len(all_chapters)}")
    print(f"Total pages: {sum(len(v) for v in all_chapters.values())}")
    print(f"{'='*50}")

    return all_chapters



if __name__ == "__main__":
    #── Option A: Single chapter ──
    single_folder = r""
    process_chapter(single_folder)
    

    
    ## ── Option B: Multiple chapters (auto-detects all subfolders) ──
    #base_dir = r""
#
    #chapter_folders = sorted([
    #    os.path.join(base_dir, d)
    #    for d in os.listdir(base_dir)
    #    if os.path.isdir(os.path.join(base_dir, d))
    #    and not d.startswith("_")   # skip _ocr_progress folders
    #])
#
    #process_all_chapters(chapter_folders, master_output_dir=base_dir)
#