import os, json
from tqdm import tqdm
from PIL import Image
import dotenv

from prompt.utils import format_ship_spatial_text
from prompt.prompt import (GENERAL_SYS_PROMPT, GENERAL_USER_PROMPT, 
                           POSITION_SYS_PROMPT, POSITION_USER_PROMPT, 
                           APPEARANCE_SYS_PROMPT)
from gemini_handler import GeminiSDKHandler

UPDATE_GENERAL = False
UPDATE_POSITION = False
UPDATE_APPEARANCE = True

dotenv.load_dotenv()

data_dir = "data/metadata/train"
rgb_dir = "data/imgs/train/rgb_images"
handler = GeminiSDKHandler(data_dir=data_dir)

seqs_all = sorted([os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')])
seqs = [s for s in seqs_all if os.path.exists(os.path.join(data_dir, f"result_{s}.json"))]

progress = tqdm(total=len(seqs), desc="Updating via Gemini SDK")
handler.set_progress_bar(progress)

for seq in seqs[:10]:
    output_path = os.path.join(data_dir, f"result_{seq}.json")
    with open(output_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    full_img = Image.open(os.path.join(rgb_dir, f"{seq}.jpg")).convert("RGB")
    W, H = full_img.size
    tasks = []

    if UPDATE_GENERAL:
        ship_info = "\n".join([f"- {s_id}: {info['class']}, {info['position']}" for s_id, info in meta["objects_enrichment"].items()])
        usr = GENERAL_USER_PROMPT.format(imaging_time=meta["metadata"]["imaging_time"], resolution=meta["metadata"]["resolution"], 
                                         center_coords=meta["metadata"]["center_coordinates"], ship_info=ship_info)
        tasks.append({'type': 'general', 'sys': GENERAL_SYS_PROMPT, 'usr': usr, 'img': full_img})

    if UPDATE_POSITION:
        spatial_text = "\n".join(format_ship_spatial_text(meta["objects_enrichment"], top_k=8).values())
        tasks.append({'type': 'position', 'sys': POSITION_SYS_PROMPT, 'usr': POSITION_USER_PROMPT.format(ship_data=spatial_text), 'img': full_img})

    if UPDATE_APPEARANCE:
        for s_id, info in meta["objects_enrichment"].items():
            xc, yc, w_norm, h_norm = info["position"]
            side = max(w_norm * W, h_norm * H) * 1.15
            patch = full_img.crop((max(0, xc*W-side/2), max(0, yc*H-side/2), min(W, xc*W+side/2), min(H, yc*H+side/2)))
            tasks.append({'type': 'appearance', 'sys': APPEARANCE_SYS_PROMPT, 'usr': "Describe this ship", 'img': patch, 'ship_id': s_id})

    handler.update_seq(seq, meta, tasks)
