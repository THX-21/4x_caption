import os, json, asyncio
from tqdm import tqdm
from PIL import Image
import dotenv

from prompt.utils import format_ship_spatial_text
from prompt.prompt import (GENERAL_SYS_PROMPT, GENERAL_USER_PROMPT, 
                           POSITION_SYS_PROMPT, POSITION_USER_PROMPT, 
                           APPEARANCE_SYS_PROMPT)
from openai_handler import OpenAIHandler

UPDATE_GENERAL = False
UPDATE_POSITION = False
UPDATE_APPEARANCE = True
MAX_CONCURRENT_TASKS = 6

dotenv.load_dotenv()

data_dir = "/root/autodl-tmp/wd/4x_caption/data/metadata/train"
rgb_dir = "/root/autodl-fs/data/imgs/train/rgb_images"
handler = OpenAIHandler(data_dir=data_dir, max_concurrent=MAX_CONCURRENT_TASKS)

seqs_all = sorted([os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')])
seqs = [s for s in seqs_all if os.path.exists(os.path.join(data_dir, f"result_{s}.json"))]

progress = tqdm(total=len(seqs), desc="Updating via OpenAI (Async)")
handler.set_progress_bar(progress)

async def main():
    all_tasks = []
    for seq in seqs[400:500]:
        try:
            output_path = os.path.join(data_dir, f"result_{seq}.json")
            with open(output_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            full_img = Image.open(os.path.join(rgb_dir, f"{seq}.jpg")).convert("RGB")
            W, H = full_img.size
            tasks_data = []

            if UPDATE_GENERAL:
                ship_info = "\n".join([f"- {s_id}: {info['class']}, {info['position']}" for s_id, info in meta["objects_enrichment"].items()])
                usr = GENERAL_USER_PROMPT.format(imaging_time=meta["metadata"]["imaging_time"], resolution=meta["metadata"]["resolution"], 
                                                 center_coords=meta["metadata"]["center_coordinates"], ship_info=ship_info)
                tasks_data.append({'type': 'general', 'sys': GENERAL_SYS_PROMPT, 'usr': usr, 'img': full_img})

            if UPDATE_POSITION:
                spatial_text = "\n".join(format_ship_spatial_text(meta["objects_enrichment"], top_k=8).values())
                tasks_data.append({'type': 'position', 'sys': POSITION_SYS_PROMPT, 'usr': POSITION_USER_PROMPT.format(ship_data=spatial_text), 'img': full_img})

            if UPDATE_APPEARANCE:
                for s_id, info in meta["objects_enrichment"].items():
                    xc, yc, w_norm, h_norm = info["position"]
                    side = max(w_norm * W, h_norm * H) * 1.15
                    patch = full_img.crop((max(0, xc*W-side/2), max(0, yc*H-side/2), min(W, xc*W+side/2), min(H, yc*H+side/2)))
                    tasks_data.append({'type': 'appearance', 'sys': APPEARANCE_SYS_PROMPT, 'usr': "Describe this ship", 'img': patch, 'ship_id': s_id})

            if not tasks_data:
                progress.update(1)
                continue

            all_tasks.append(handler.update_seq(seq, meta, tasks_data))
                
        except Exception as e:
            print(f"\nError preparing {seq}: {e}")
            progress.update(1)
    
    if all_tasks:
        await asyncio.gather(*all_tasks)

if __name__ == "__main__":
    asyncio.run(main())
