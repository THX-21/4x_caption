import os
import json
import json_repair
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

class VLLMTaskHandler:
    def __init__(self, model_id="Qwen/Qwen3-VL-32B-Instruct", chunk_size=8, gpu_memory_utilization=0.9):
        os.environ["VLLM_USE_MODELSCOPE"] = "False"
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.llm = LLM(
            model=model_id, 
            limit_mm_per_prompt={"image": 1}, 
            max_model_len=9216, 
            gpu_memory_utilization=gpu_memory_utilization, 
            swap_space=24, 
            enable_chunked_prefill=True, 
            max_num_batched_tokens=2048
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.sampling_params = SamplingParams(max_tokens=2048, temperature=0.01)
        
        self.all_metas = {}
        self.tasks_remaining = {}
        self.buffers = {"general": [], "position": [], "appearance": []}
        self.progress = None

    def set_progress_bar(self, progress_bar):
        self.progress = progress_bar

    def make_input(self, sys_p, usr_p, img):
        msg = [
            {"role": "system", "content": [{"type": "text", "text": sys_p}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": usr_p}]}
        ]
        return {
            "prompt": self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True),
            "multi_modal_data": {"image": img}
        }

    def save_result(self, seq):
        meta = self.all_metas.get(seq)
        if not meta: return
        output_path = f"data/metadata/result_{seq}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        print(f"Saved: {output_path}")
        if self.progress is not None:
            self.progress.update(1)

    def run_batch(self, buffer_type):
        buffer = self.buffers[buffer_type]
        if not buffer: return
        
        try:
            outputs = self.llm.generate([t["input"] for t in buffer], self.sampling_params)
            for task, out in zip(buffer, outputs):
                seq = task["seq"]
                try:
                    res_text = out.outputs[0].text
                    data = json_repair.loads(res_text)
                    meta = self.all_metas[seq]
                    
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
                    
                    self.tasks_remaining[seq] -= 1
                    if self.tasks_remaining[seq] == 0:
                        self.save_result(seq)
                except Exception as e:
                    print(f"Error parsing LLM output for {seq}: {e}")
        except Exception as e:
            print(f"CRITICAL: llm.generate failed for batch: {e}")
            for task in buffer:
                print(f"Error processing {task['seq']}: {e}")
        
        buffer.clear()

    def add_task(self, task):
        b_type = task["type"]
        self.buffers[b_type].append(task)
        if len(self.buffers[b_type]) >= self.chunk_size:
            self.run_batch(b_type)

    def flush_all(self):
        for b_type in self.buffers:
            self.run_batch(b_type)
