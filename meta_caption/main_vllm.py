from prompt.utils import extract_normalized_info, format_ship_spatial_text
from prompt.prompt import (GENERAL_SYS_PROMPT, GENERAL_USER_PROMPT, 
                           POSITION_SYS_PROMPT, POSITION_USER_PROMPT, 
                           APPEARANCE_SYS_PROMPT)


import os, json
from tqdm import tqdm


os.environ["VLLM_USE_MODELSCOPE"] = "False" # 如果是模型加载问题

from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import json_repair


# 1. 初始化 vLLM (针对 Qwen2/3-VL)
model_id = "Qwen/Qwen3-VL-32B-Instruct"
# limit_mm_per_prompt 设置每条 prompt 处理 1 张图
llm = LLM(model=model_id, limit_mm_per_prompt={"image": 1}, max_model_len=9216,gpu_memory_utilization=0.9, 
    swap_space=24,enable_chunked_prefill=True, max_num_batched_tokens=2048)
processor = AutoProcessor.from_pretrained(model_id)
sampling_params = SamplingParams(max_tokens=2048, temperature=0.01)

os.makedirs("data", exist_ok=True)

# gather all image sequence ids from rgb_images folder
rgb_dir = "/root/autodl-fs/RGB/test/rgb_images"
try:
    seqs_all = sorted([os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')])
except Exception as e:
    print(f"Could not list rgb images in {rgb_dir}: {e}")
    seqs_all = []

# Only process sequences that don't already have a result file in data/
seqs = [s for s in seqs_all if not os.path.exists(os.path.join("data", f"result_{s}.json"))]
if not seqs:
    print("No new images to process (all results already exist in data/).")

# 存储所有元数据引用
all_metas = {}
tasks_remaining = {}
progress = None

def save_result(seq):
    meta = all_metas.get(seq)
    if not meta: return
    output_path = f"data/result_{seq}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"Saved: {output_path}")
    try:
        if progress is not None:
            progress.update(1)
    except Exception:
        pass

# 2. 任务执行器
CHUNK_SIZE = 8
buffers = {"general": [], "position": [], "appearance": []}

def run_batch(buffer):
    if not buffer: return
    try:
        outputs = llm.generate([t["input"] for t in buffer], sampling_params)
        for task, out in zip(buffer, outputs):
            seq = task["seq"]
            try:
                res_text = out.outputs[0].text
                data = json_repair.loads(res_text)
                meta = all_metas[seq]
                if task["type"] == "position":
                    for item in data:
                        s_id = item.get("ship_id")
                        if s_id in meta["objects_enrichment"]:
                            meta["objects_enrichment"][s_id]["immediate_surroundings"] = item.get("immediate_surroundings", "")
                elif task["type"] == "general":
                    if "scene_context" in data:
                        meta["scene_context"].update(data["scene_context"])
                    if "objects_enrichment" in data:
                        for s_id, ob in data["objects_enrichment"].items():
                            if s_id in meta["objects_enrichment"]:
                                meta["objects_enrichment"][s_id].update(ob)
                else:
                    meta["objects_enrichment"][task["ship_id"]]["visual_appearance"] = data.get("visual_appearance", "")
                tasks_remaining[seq] -= 1
                if tasks_remaining[seq] == 0:
                    save_result(seq)
            except Exception as e: 
                print(f"Error parsing LLM output for {seq}: {e}")
            

    except Exception as e:
        print(f"CRITICAL: llm.generate failed for batch: {e}")
        import traceback
        traceback.print_exc()
        # 如果整批失败，也要更新计数器，防止这些 seq 永远无法保存
        for task in buffer:
            seq = task["seq"]
            print(f"Error processing {seq}: {e}")
            
    buffer.clear()

def add_task(task):
    b = buffers[task["type"]]
    b.append(task)
    if len(b) >= CHUNK_SIZE: run_batch(b)

progress = tqdm(total=len(seqs), desc="Processing images")
for seq in seqs:
    print(f"--- Processing {seq} ---")
    tif, lbl, jpg = f"/root/autodl-fs/RGB/test/images/{seq}.tif", f"/root/autodl-fs/RGB/test/labels/{seq}.json", f"/root/autodl-fs/RGB/test/rgb_images/{seq}.jpg"
    if not os.path.exists(jpg): continue
    
    full_img = Image.open(jpg).convert("RGB")
    W, H = full_img.size
    meta = extract_normalized_info(tif, lbl)
    all_metas[seq] = meta
    
    # 记录该 seq 总任务数：1(general) + 1(position) + n(appearance)
    tasks_remaining[seq] = 2 + len(meta["objects_enrichment"])

    def make_input(sys_p, usr_p, img):
        msg = [{"role": "system", "content": [{"type": "text", "text": sys_p}]},
               {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": usr_p}]}]
        return {"prompt": processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True),
                "multi_modal_data": {"image": img}}

    # 1. General 描述任务 (仅针对 scene_context)
    ship_info_text = "\n".join([f"- {s_id}: {info['class']}, {info['position']}" for s_id, info in meta["objects_enrichment"].items()])
    general_user_content = GENERAL_USER_PROMPT.format(
        imaging_time=meta["metadata"]["imaging_time"],
        resolution=meta["metadata"]["resolution"],
        center_coords=meta["metadata"]["center_coordinates"],
        ship_info=ship_info_text
    )
    add_task({"type": "general", "seq": seq, "input": make_input(GENERAL_SYS_PROMPT, general_user_content, full_img)})
    
    # 2. 空间描述任务 (LLM 一起生成，量化文本分别储存)
    spatial_dict_for_meta = format_ship_spatial_text(meta["objects_enrichment"], top_k=20)
    for s_id, s_text in spatial_dict_for_meta.items():
        meta["objects_enrichment"][s_id]["spatial_context"] = s_text
    spatial_dict_for_llm = format_ship_spatial_text(meta["objects_enrichment"], top_k=8)
    full_spatial_text = "\n".join(spatial_dict_for_llm.values())
    add_task({"type": "position", "seq": seq, "input": make_input(POSITION_SYS_PROMPT, POSITION_USER_PROMPT.format(ship_data=full_spatial_text), full_img)})

    for s_id, info in meta["objects_enrichment"].items():
        xc, yc, w_norm, h_norm = info["position"]
        side = max(w_norm * W, h_norm * H) * 1.15
        patch = full_img.crop((max(0, xc*W-side/2), max(0, yc*H-side/2), min(W, xc*W+side/2), min(H, yc*H+side/2)))
        if max(patch.size) < 224:
            scale = 224 / max(patch.size)
            patch = patch.resize((int(patch.width*scale), int(patch.height*scale)), Image.LANCZOS)
        add_task({"type": "appearance", "seq": seq, "ship_id": s_id, "input": make_input(APPEARANCE_SYS_PROMPT, "Describe this ship", patch)})

for b in buffers.values(): run_batch(b)
try:
    progress.close()
except Exception:
    pass