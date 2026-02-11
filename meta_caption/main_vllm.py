from prompt.utils import extract_normalized_info, format_ship_spatial_text
from prompt.prompt import (GENERAL_SYS_PROMPT, GENERAL_USER_PROMPT, 
                           POSITION_SYS_PROMPT, POSITION_USER_PROMPT, 
                           APPEARANCE_SYS_PROMPT)


import os, json
from tqdm import tqdm
from PIL import Image
from vllm_handler import VLLMTaskHandler

# 1. 初始化任务处理器
data_dir = "data/metadata/train"
handler = VLLMTaskHandler(data_dir=data_dir)

# gather all image sequence ids from rgb_images folder
rgb_dir = "/root/autodl-fs/data/imgs/train/rgb_images"
try:
    seqs_all = sorted([os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')])
except Exception as e:
    print(f"Could not list rgb images in {rgb_dir}: {e}")
    seqs_all = []

# Only process sequences that don't already have a result file in data/
seqs = [s for s in seqs_all if not os.path.exists(os.path.join(data_dir, f"result_{s}.json"))]
if not seqs:
    print("No new images to process (all results already exist in data/).")

progress = tqdm(total=len(seqs), desc="Processing images")
handler.set_progress_bar(progress)

for seq in seqs:
    print(f"--- Processing {seq} ---")
    tif, lbl, jpg = f"/root/autodl-fs/data/imgs/train/images/{seq}.tif", f"/root/autodl-fs/data/imgs/train/labels/{seq}.json", f"/root/autodl-fs/data/imgs/train/rgb_images/{seq}.jpg"
    if not os.path.exists(jpg): continue
    
    full_img = Image.open(jpg).convert("RGB")
    W, H = full_img.size
    meta = extract_normalized_info(tif, lbl)
    handler.all_metas[seq] = meta
    
    # 记录该 seq 总任务数：1(general) + 1(position) + n(appearance)
    handler.tasks_remaining[seq] = 2 + len(meta["objects_enrichment"])

    # 1. General 描述任务 (仅针对 scene_context)
    ship_info_text = "\n".join([f"- {s_id}: {info['class']}, {info['position']}" for s_id, info in meta["objects_enrichment"].items()])
    general_user_content = GENERAL_USER_PROMPT.format(
        imaging_time=meta["metadata"]["imaging_time"],
        resolution=meta["metadata"]["resolution"],
        center_coords=meta["metadata"]["center_coordinates"],
        ship_info=ship_info_text
    )
    handler.add_task({"type": "general", "seq": seq, "input": handler.make_input(GENERAL_SYS_PROMPT, general_user_content, full_img)})
    
    # 2. 空间描述任务 (LLM 一起生成，量化文本分别储存)
    spatial_dict_for_meta = format_ship_spatial_text(meta["objects_enrichment"], top_k=20)
    for s_id, s_text in spatial_dict_for_meta.items():
        meta["objects_enrichment"][s_id]["spatial_context"] = s_text
    spatial_dict_for_llm = format_ship_spatial_text(meta["objects_enrichment"], top_k=8)
    full_spatial_text = "\n".join(spatial_dict_for_llm.values())
    handler.add_task({"type": "position", "seq": seq, "input": handler.make_input(POSITION_SYS_PROMPT, POSITION_USER_PROMPT.format(ship_data=full_spatial_text), full_img)})

    for s_id, info in meta["objects_enrichment"].items():
        xc, yc, w_norm, h_norm = info["position"]
        side = max(w_norm * W, h_norm * H) * 1.15
        patch = full_img.crop((max(0, xc*W-side/2), max(0, yc*H-side/2), min(W, xc*W+side/2), min(H, yc*H+side/2)))
        if max(patch.size) < 224:
            scale = 224 / max(patch.size)
            patch = patch.resize((int(patch.width*scale), int(patch.height*scale)), Image.LANCZOS)
        handler.add_task({"type": "appearance", "seq": seq, "ship_id": s_id, "input": handler.make_input(APPEARANCE_SYS_PROMPT, "Describe this ship", patch)})

handler.flush_all()
try:
    progress.close()
except Exception:
    pass