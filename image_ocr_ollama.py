import ollama
import base64
import json
import os
import re
from tqdm import tqdm

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
                            You are a Manhwa/Webtoon Analyst and Transcriber. Your job is to analyze panels from a manhwa chapter and extract maximum   textual and visual information. 
                        Very Very Important: Return ONLY the JSON object. No extra text. Keys must be exactly: page_type, mood_and_atmosphere,  visual_description, transcribed_text, narrator_hints.
                        Do not use markdown formatting outside of the JSON block. Do not hallucinate text or events that are not in the image.



                            Analyze the image and adapt your extraction based on the PAGE TYPE:

                            1. "EVERYTHING" (Standard Pages): 
                               - Transcribe all speech bubbles, narration boxes, and thought bubbles.

                               - Identify who is speaking (if unknown, describe them via their distinct features: eg. "Red-Haired Man").
                               - Describe character actions, facial expressions, and spatial positioning.
                            2. "IMAGES_ONLY" (Action/Scenery, No Text):
                               - Focus entirely on the choreography of the action, kinetic energy, or the beauty of the scenery.

                               - Describe character emotions, weapon movements, impact points, and camera angles (e.g., "low angle looking up at the        monster").
                        3. "TEXT_ONLY" (Blank backgrounds with text):
                               - Transcribe the text with 100% accuracy.

                               - Note the background color and the visual styling of the text (e.g., "White text on a pitch-black background, jagged font   indicating screaming").
                        4. "ARTISTIC_SPREAD" (Long vertical art piece / Epic reveals):
                               - Read the image from TOP to BOTTOM.

                               - Describe the scale, lighting, mood, and atmosphere. Focus heavily on the awe-inspiring or dramatic elements.
                            REQUIRED JSON SCHEMA:
                            Output in this manner:

                            {
                              "page_type": "standard | images_only | text_only | artistic_spread | photo_insert",

                              "mood_and_atmosphere": "Brief description of the lighting, colors, and emotional tone",
                              "visual_description": "Detailed, chronological (top-to-bottom) description of what is happening visually. If artistic/    action, be highly descriptive.",
                          "transcribed_text": [
                                {
                                  "type": "speech | thought | narration | sound_effect | floating_text",
                                  "speaker_description": "Who is saying this? (or 'None' if narration/SFX)",
                                  "text": "The exact transcribed text"
                                }
                              ],
                              "narrator_hints": "Suggestions for the narrator AI (e.g., 'Long pause needed here', 'Fast-paced action scene', 'Ominous   reveal')"
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



if __name__ == "__main__":
    folder_path=r""
    data=get_file_paths(folder_path)

    print(json.dumps(data, indent=2, ensure_ascii=False))

    with open(os.path.join(folder_path, "chapter_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        print("Data successfully saved to chapter_metadata.json!")
