from prompt.utils import extract_normalized_info, format_ship_spatial_text
from prompt.prompt import (GENERAL_SYS_PROMPT, GENERAL_USER_PROMPT, 
                           POSITION_SYS_PROMPT, POSITION_USER_PROMPT, 
                           APPEARANCE_SYS_PROMPT)

import os, json, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import json_repair


# 初始化模型与处理器
model_id = "Qwen/Qwen3-VL-32B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="auto", device_map="auto")

def ask_qwen(system_prompt, user_content, pil_image):
    """优化后的 Qwen 调用接口"""
    # 确保格式统一：system 和 user 都采用 list of dict 格式
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_content}
            ]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[pil_image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, out)
    ]
    
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

# 确保输出目录存在
os.makedirs("data", exist_ok=True)
seqs = ["00001", "00011", "00021", "00041", "00051", "00061", "00081", "00101"] # 待处理序列
for seq in seqs:
    print(f"--- Processing {seq} ---")
    tif, lbl, jpg = f"/root/autodl-fs/RGB/train/images/{seq}.tif", f"/root/autodl-fs/RGB/train/labels/{seq}.json", f"/root/autodl-fs/RGB/train/rgb_images/{seq}.jpg"
    if not os.path.exists(jpg):
        print(f"File {jpg} not found, skipping...")
        continue
        
    full_img = Image.open(jpg).convert("RGB")
    W, H = full_img.size

    # 1. 获取 Meta 数据
    meta = extract_normalized_info(tif, lbl)

    # 2. General 描述 (全图场景 - 仅针对 scene_context)
    ship_info_text = "\n".join([f"- {s_id}: {info['class']}, {info['position']}" for s_id, info in meta["objects_enrichment"].items()])
    general_user_content = GENERAL_USER_PROMPT.format(
        imaging_time=meta["metadata"]["imaging_time"],
        resolution=meta["metadata"]["resolution"],
        center_coords=meta["metadata"]["center_coordinates"],
        ship_info=ship_info_text
    )
    gen_res = ask_qwen(GENERAL_SYS_PROMPT, general_user_content, full_img)
    try:
        gen_data = json_repair.loads(gen_res)
        if "scene_context" in gen_data:
            meta["scene_context"].update(gen_data["scene_context"])
        if "objects_enrichment" in gen_data:
            for s_id, ob in gen_data["objects_enrichment"].items():
                if s_id in meta["objects_enrichment"]:
                    meta["objects_enrichment"][s_id].update(ob)
    except Exception as e: print(f"General parse error for {seq}: {e}")

    # 3. 空间位置描述 (LLM 一起生成，量化文本分别储存)
    spatial_dict = format_ship_spatial_text(meta["objects_enrichment"])
    for s_id, s_text in spatial_dict.items():
        meta["objects_enrichment"][s_id]["spatial_context"] = s_text
    full_spatial_text = "\n".join(spatial_dict.values())
    pos_res = ask_qwen(POSITION_SYS_PROMPT, POSITION_USER_PROMPT.format(ship_data=full_spatial_text), full_img)
    try:
        pos_data = json_repair.loads(pos_res)
        for item in pos_data:
            s_id = item["ship_id"]
            immediate_surroundings = item["immediate_surroundings"]
            if s_id in meta["objects_enrichment"]:
                meta["objects_enrichment"][s_id]["immediate_surroundings"] = immediate_surroundings
    except: print(f"Position parse error for {seq}")

    # 4. 切片外貌描述 (采用插值放大，保持主体地位)
    MIN_PATCH_SIZE = 224
    for s_id, info in meta["objects_enrichment"].items():
        xc, yc, w_norm, h_norm = info["position"]
        pixel_xc, pixel_yc = xc * W, yc * H
        pixel_w, pixel_h = w_norm * W, h_norm * H
        
        # 仅保留 15% 的 Padding，使舰船占据主体
        side_len = max(pixel_w, pixel_h) * 1.15
        box = (
            max(0, pixel_xc - side_len/2),
            max(0, pixel_yc - side_len/2),
            min(W, pixel_xc + side_len/2),
            min(H, pixel_yc + side_len/2)
        )
        patch = full_img.crop(box)
        
        # 如果切片小于阈值，插值放大到 MIN_PATCH_SIZE
        if patch.width < MIN_PATCH_SIZE or patch.height < MIN_PATCH_SIZE:
            scale = MIN_PATCH_SIZE / max(patch.width, patch.height)
            new_size = (int(patch.width * scale), int(patch.height * scale))
            patch = patch.resize(new_size, Image.LANCZOS)
        
        app_res = ask_qwen(APPEARANCE_SYS_PROMPT, f"Describe this ship", patch)
        try:
            content = json_repair.loads(app_res)
            meta["objects_enrichment"][s_id]["visual_appearance"] = content.get("visual_appearance", "")
        except: print(f"Appearance parse error for {s_id}")

    # 保存结果
    output_path = f"data/result_{seq}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"Saved: {output_path}")
