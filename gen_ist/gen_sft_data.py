import json
import json_repair
import os
import random
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt_en.image_caption import CaptionTemplateEngine
from prompt_en.object_detection import DetectionTemplateEngine
from prompt_en.visual_grounding import VisualGroundingTemplateEngine
from prompt_en.VQA import VQATemplateEngine
from prompt_en.conversation import ConversationTemplateEngine

# 加载环境变量
load_dotenv()

# 配置 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

class SFTDataGenerator:
    def __init__(self, class_map):
        self.class_map = class_map
        self.caption_engine = CaptionTemplateEngine()
        self.det_engine = DetectionTemplateEngine()
        self.vg_engine = VisualGroundingTemplateEngine()
        self.vqa_engine = VQATemplateEngine()
        self.conv_engine = ConversationTemplateEngine()

    def call_llm(self, sys_pt, user_pt):
        """调用 OpenAI 接口"""
        # 在系统提示词中明确告知模型：以 ship_type_{i} 格式引用船只类别
        enhanced_sys_pt = sys_pt + "\n\nNote: The categories are provided as 'ship_type_{i}' (e.g., 'ship_type_37', 'ship_type_10'). Please use this exact format 'ship_type_{i}' directly in your descriptions and answers whenever referring to a ship's category."
        try:
            response = client.chat.completions.create(
                # model="deepseek-chat",
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": enhanced_sys_pt},
                    {"role": "user", "content": user_pt}
                ],
                response_format={"type": "json_object"},
                stream=False
            )
            return json_repair.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None

    def process_image_data(self, image_id, raw_data):
        """处理单张图像数据，生成不同任务的 SFT 数据"""
        sft_results = []

        # 1. 生成 Caption 数据
        sys_pt, user_pt = self.caption_engine.get_prompts(self.caption_engine.flatten_data(raw_data))
        caption_res = self.call_llm(sys_pt, user_pt)
        if caption_res:
            for style, content in caption_res.items():
                sft_results.append({
                    "id": f"{image_id}_caption_{style}",
                    "image": f"{image_id}.jpg",
                    "conversations": [
                        {"from": "human", "value": content["Instruction"]},
                        {"from": "gpt", "value": content["Answer"]}
                    ]
                })

        # 2. Object Detection (规则生成，不需要 LLM)
        det_data = self.det_engine.generate_data(raw_data["objects_enrichment"], self.class_map)
        for i, item in enumerate(det_data):
            sft_results.append({
                "id": f"{image_id}_det_{i}",
                "image": f"{image_id}.jpg",
                "conversations": [
                    {"from": "human", "value": item["instruction"]},
                    {"from": "gpt", "value": item["answer"]}
                ]
            })

        # 3. Visual Grounding (调用 LLM)
        sys_pt, user_pt = self.vg_engine.get_prompts(raw_data, self.class_map)
        vg_res = self.call_llm(sys_pt, user_pt) # 返回的是列表
        if isinstance(vg_res, list):
            for i, item in enumerate(vg_res):
                sft_results.append({
                    "id": f"{image_id}_vg_{i}",
                    "image": f"{image_id}.jpg",
                    "conversations": [
                        {"from": "human", "value": item["Instruction"]},
                        {"from": "gpt", "value": str(item["Answer"])}
                    ]
                })
        elif isinstance(vg_res, dict) and "Visual_Grounding" in vg_res: # 兼容某些返回格式
             for i, item in enumerate(vg_res["Visual_Grounding"]):
                sft_results.append({
                    "id": f"{image_id}_vg_{i}",
                    "image": f"{image_id}.jpg",
                    "conversations": [
                        {"from": "human", "value": item["Instruction"]},
                        {"from": "gpt", "value": str(item["Answer"])}
                    ]
                })

        # 4. VQA (调用 LLM)
        sys_pt, user_pt = self.vqa_engine.get_prompts(raw_data, self.class_map)
        vqa_res = self.call_llm(sys_pt, user_pt)
        if isinstance(vqa_res, list):
            for i, item in enumerate(vqa_res):
                sft_results.append({
                    "id": f"{image_id}_vqa_{i}",
                    "image": f"{image_id}.jpg",
                    "conversations": [
                        {"from": "human", "value": item["Instruction"]},
                        {"from": "gpt", "value": item["Answer"]}
                    ]
                })

        # 5. Conversation (调用 LLM)
        sys_pt, user_pt = self.conv_engine.get_prompts(raw_data, self.class_map)
        conv_res = self.call_llm(sys_pt, user_pt)
        if conv_res and "Conversation" in conv_res:
            conv_list = []
            for turn in conv_res["Conversation"]:
                conv_list.append({"from": "human", "value": turn["user"]})
                conv_list.append({"from": "gpt", "value": turn["assistant"]})
            sft_results.append({
                "id": f"{image_id}_conv",
                "image": f"{image_id}.jpg",
                "conversations": conv_list
            })

        return sft_results

if __name__ == "__main__":
    my_class_map = {str(i): f"ship_type_{i}" for i in range(100)}
    generator = SFTDataGenerator(my_class_map)
    
    train_dir = "/root/autodl-fs/data/metadata/train"
    out_dir = "data/train"
    os.makedirs(out_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(train_dir) if f.endswith(".json")]
    json_files = json_files[50:100]
    
    to_process = sorted([f for f in json_files if not os.path.exists(os.path.join(out_dir, f))])
    def worker(filename):
        out_path = os.path.join(out_dir, filename)
        with open(os.path.join(train_dir, filename), "r") as f:
            raw_data = json.load(f)
        res = generator.process_image_data(filename.replace(".json", ""), raw_data)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
        return filename
    
    DEBUG = True # 调试代码：串行调用
    if to_process:
        if DEBUG:
            for fn in tqdm(to_process, desc="生成 SFT 数据 (串行调试)"):
                worker(fn)
        else:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(worker, fn) for fn in to_process]
                pbar = tqdm(total=len(futures), desc="生成 SFT 数据")
                for _ in as_completed(futures):
                    pbar.update(1)
                pbar.close()
