import os, json, base64, io, json_repair
from google import genai
from google.genai import types
from PIL import Image

class GeminiSDKHandler:
    def __init__(self, api_key=None, base_url="https://api.zhizengzeng.com/google", model="gemini-3-flash-preview", data_dir="data/metadata/train"):
        self.client = genai.Client(
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            http_options={"base_url": base_url or os.getenv("GOOGLE_BASE_URL")}
        )
        self.model = model or os.getenv("GOOGLE_MODEL")
        self.data_dir = data_dir
        self.progress = None

    def set_progress_bar(self, progress_bar):
        self.progress = progress_bar

    def call_gemini(self, sys_p, usr_p, img):
        try:
            # 将 PIL Image 转换为 bytes
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            image_bytes = buffered.getvalue()

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                    f"{sys_p}\n\n{usr_p}"
                ],
                config=types.GenerateContentConfig(
                    temperature=0.01,
                    response_mime_type="application/json"
                )
            )
            return json_repair.loads(response.text)
        except Exception as e:
            print(f"Error calling Gemini SDK: {e}")
            return {}

    def update_seq(self, seq, meta, tasks):
        try:
            for task in tasks:
                res = self.call_gemini(task['sys'], task['usr'], task['img'])
                if task['type'] == 'general':
                    meta.get("scene_context", {}).update(res.get("scene_context", {}))
                elif task['type'] == 'position':
                    for item in (res if isinstance(res, list) else []):
                        s_id = item.get("ship_id")
                        if s_id in meta["objects_enrichment"]:
                            meta["objects_enrichment"][s_id]["immediate_surroundings"] = item.get("immediate_surroundings", "")
                elif task['type'] == 'appearance':
                    meta["objects_enrichment"][task['ship_id']]["visual_appearance"] = res.get("visual_appearance", "")
            
            with open(os.path.join(self.data_dir, f"result_{seq}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error processing {seq}: {e}")
        if self.progress: self.progress.update(1)
