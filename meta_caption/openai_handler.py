import os
import json
import base64
import io
import asyncio
import json_repair
from openai import AsyncOpenAI
from PIL import Image

class OpenAIHandler:
    def __init__(self, api_key=None, base_url=None, model=None, data_dir="data/metadata/train", max_concurrent=10):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))
        self.model = model or os.getenv("OPENAI_MODEL")
        self.data_dir = data_dir
        self.all_metas = {}
        self.progress = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def set_progress_bar(self, progress_bar):
        self.progress = progress_bar

    def encode_image(self, img):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def call_openai(self, sys_p, usr_p, img):
        async with self.semaphore:
            base64_image = self.encode_image(img)
            response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": [
                    {"type": "text", "text": usr_p},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.01,
            response_format={"type": "json_object"}
        )
        return json_repair.loads(response.choices[0].message.content)

    def apply_task_result(self, task, meta, res):
        if task['type'] == 'general':
            meta.get("scene_context", {}).update(res.get("scene_context", {}))
        elif task['type'] == 'position':
            for item in res if isinstance(res, list) else []:
                s_id = item.get("ship_id")
                if s_id in meta["objects_enrichment"]:
                    meta["objects_enrichment"][s_id]["immediate_surroundings"] = item.get("immediate_surroundings", "")
        elif task['type'] == 'appearance':
            meta["objects_enrichment"][task['ship_id']]["visual_appearance"] = res.get("visual_appearance", "")

    async def process_single_task(self, task, meta):
        res = await self.call_openai(task['sys'], task['usr'], task['img'])
        self.apply_task_result(task, meta, res)
        return res

    async def update_seq(self, seq, meta, tasks):
        try:
            # 并发执行所有子任务
            coros = [self.call_openai(task['sys'], task['usr'], task['img']) for task in tasks]
            results = await asyncio.gather(*coros)
            
            for task, res in zip(tasks, results):
                self.apply_task_result(task, meta, res)
            
            output_path = os.path.join(self.data_dir, f"result_{seq}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error processing {seq}: {e}")
        if self.progress: self.progress.update(1)
