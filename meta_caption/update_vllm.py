import os, json
from tqdm import tqdm
from PIL import Image
from prompt.utils import format_ship_spatial_text
from prompt.prompt import (GENERAL_SYS_PROMPT, GENERAL_USER_PROMPT, 
                           POSITION_SYS_PROMPT, POSITION_USER_PROMPT, 
                           APPEARANCE_SYS_PROMPT)
from vllm_handler import VLLMTaskHandler

# ==========================================
# 更新配置 (设置为 True 以重新运行对应任务)
# ==========================================
UPDATE_GENERAL = False    # 更新场景上下文 (scene_context)
UPDATE_POSITION = False   # 更新空间环境 (immediate_surroundings)
UPDATE_APPEARANCE = True  # 更新视觉外观 (visual_appearance)
# ==========================================

# 1. 初始化任务处理器
data_dir = "data/metadata/train"
handler = VLLMTaskHandler(data_dir=data_dir)

rgb_dir = "/root/autodl-fs/RGB/test/rgb_images"
try:
    seqs_all = sorted([os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')])
except Exception as e:
    print(f"Could not list rgb images in {rgb_dir}: {e}")
    seqs_all = []

# 仅处理已有结果的文件进行更新
seqs = [s for s in seqs_all if os.path.exists(os.path.join(data_dir, f"result_{s}.json"))]

if not seqs:
    print("No existing result files found in data/ to update.")
    exit()

progress = tqdm(total=len(seqs), desc="Updating metadata")
handler.set_progress_bar(progress)

for seq in seqs:
    output_path = os.path.join(data_dir, f"result_{seq}.json")
    jpg = f"/root/autodl-fs/RGB/test/rgb_images/{seq}.jpg"
    
    with open(output_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    handler.all_metas[seq] = meta
    
    # 计算需要运行的任务数
    task_count = 0
    if UPDATE_GENERAL: task_count += 1
    if UPDATE_POSITION: task_count += 1
    if UPDATE_APPEARANCE: task_count += len(meta["objects_enrichment"])
    
    handler.tasks_remaining[seq] = task_count
    
    if task_count == 0:
        print(f"No update tasks selected for {seq}, skipping.")
        progress.update(1)
        continue

    full_img = Image.open(jpg).convert("RGB")
    W, H = full_img.size

    # 1. General 描述任务
    if UPDATE_GENERAL:
        ship_info_text = "\n".join([f"- {s_id}: {info['class']}, {info['position']}" for s_id, info in meta["objects_enrichment"].items()])
        general_user_content = GENERAL_USER_PROMPT.format(
            imaging_time=meta["metadata"]["imaging_time"],
            resolution=meta["metadata"]["resolution"],
            center_coords=meta["metadata"]["center_coordinates"],
            ship_info=ship_info_text
        )
        handler.add_task({"type": "general", "seq": seq, "input": handler.make_input(GENERAL_SYS_PROMPT, general_user_content, full_img)})
    
    # 2. 空间描述任务
    if UPDATE_POSITION:
        spatial_dict_for_llm = format_ship_spatial_text(meta["objects_enrichment"], top_k=8)
        full_spatial_text = "\n".join(spatial_dict_for_llm.values())
        handler.add_task({"type": "position", "seq": seq, "input": handler.make_input(POSITION_SYS_PROMPT, POSITION_USER_PROMPT.format(ship_data=full_spatial_text), full_img)})

    # 3. 视觉外观描述任务
    if UPDATE_APPEARANCE:
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
