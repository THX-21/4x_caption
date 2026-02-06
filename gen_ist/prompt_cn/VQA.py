import json

class VQATemplateEngine:
    def __init__(self):
        # 系统提示词：设定专家角色、任务配比及语境约束
        self.sys_pt = """### 角色：遥感影像研判专家
你是一名精通卫星遥感与航拍影像情报分析的专家。你的任务是根据提供的结构化影像资料，构建一组高质量的指令微调视觉问答对（VQA）。

### 任务核心要求：
1. **提问维度多样性**：
   - **具体信息问答（约75%）**：针对分辨率、成像时间段（早晨/下午等）、天气、特定目标的视觉特征（如颜色、标识、桅杆）、背景建筑、精确空间关系（如左上、距离）等进行提问。
   - **综合推理分析（约25%）**：结合多个字段推断任务背景（如：港口繁忙程度分析、舰队编队逻辑、环境对行动的影响等）。

2. **撰写准则**：
   - **去结构化叙述**：严禁出现“根据数据”、“字段显示”等字眼。问答应如同在查看真实照片。
   - **专业术语**：恰当使用“地物轮廓”、“几何结构”、“大气透明度”、“泊位利用率”、“拓扑关联”等词。
   - **去标识化**：提问时严禁使用 Ship_001 等编号，必须通过空间位置或视觉特征指代目标。
   - **严禁术语**：输出内容中严禁出现“元数据”或“JSON”字样。

### 输出格式要求：
必须以纯 JSON 列表格式返回，每个元素包含 Instruction 和 Answer。格式如下：
[
    {{"Instruction": "具体信息提问...", "Answer": "基于事实的回答..."}},
    {{"Instruction": "综合推理提问...", "Answer": "基于逻辑的分析..."}}
]
"""

        self.user_pt_template = """### 影像原始资料：
- **基本属性**：地面分辨率为 {resolution}，中心坐标 {coordinates}。
- **环境背景**：场景为 {scene_type}。成像时间：{time_of_day}。当时天气：{weather}。
- **背景元素**：周边存在 {background_elements}。
- **宏观布局**：{arrangement}
- **详细解译描述**：{detail_description}
- **目标个体特征与空间逻辑**：
{objects_info_block}

### 执行任务：
请根据上述资料，生成一组（共 10-12 个）专业问答对，确保涵盖具体细节询问与适量的综合推理分析。
"""

    def _format_objects_info(self, objects_enrichment, class_map):
        """将对象信息处理成更易读的文本块，隐藏内部ID，转换类别名"""
        info_blocks = []
        # 创建 ID 到编号的映射，例如 {"Ship_001": "1号船"}
        id_to_num = {obj_id: f"{i+1}号船" for i, obj_id in enumerate(objects_enrichment.keys())}
        
        for obj_id, info in objects_enrichment.items():
            c_name = class_map.get(info['class'], "未知舰船")
            # 替换 spatial_context 中的内部 ID
            spatial_info = info['spatial_context'].strip()
            # 按照 ID 长度降序排序，防止 Ship_0011 错误匹配到 Ship_001
            sorted_ids = sorted(id_to_num.keys(), key=len, reverse=True)
            for old_id in sorted_ids:
                spatial_info = spatial_info.replace(old_id, id_to_num[old_id])
            
            desc = (
                f"- {id_to_num[obj_id]}（{c_name}）：外观特征为“{info['visual_appearance']}”；"
                f"当前状态为“{info['activity_status']}”；"
                f"位置关联信息为：{spatial_info}"
            )
            info_blocks.append(desc)
        return "\n".join(info_blocks)

    def get_prompts(self, raw_json, class_map):
        """
        生成发送给 LLM 的最终系统和用户提示词
        """
        meta = raw_json.get("metadata", {})
        scene = raw_json.get("scene_context", {})
        objs = raw_json.get("objects_enrichment", {})

        # 预处理字段，避免出现“元数据”字样
        objects_info_block = self._format_objects_info(objs, class_map)
        
        user_content = self.user_pt_template.format(
            resolution=meta.get("resolution"),
            coordinates=f"{meta.get('center_coordinates', {}).get('latitude')}N, {meta.get('center_coordinates', {}).get('longitude')}E",
            scene_type=scene.get("scene_type"),
            time_of_day=scene.get("time_of_day"),
            weather=scene.get("weather_conditions"),
            background_elements=", ".join(scene.get("background_elements", [])),
            arrangement=scene.get("arrangement"),
            detail_description=scene.get("detail_description"),
            objects_info_block=objects_info_block
        )

        return self.sys_pt, user_content

# --- 模拟执行 ---
if __name__ == "__main__":
    # 模拟你的原始数据结构
    raw_data = {
        "metadata": {
            "imaging_time": "20240226",
            "resolution": "0.8m",
            "center_coordinates": {"longitude": 139.653192, "latitude": 35.289492}
        },
        "scene_context": {
            "scene_type": "Naval Base",
            "weather_conditions": "Clear skies; calm sea",
            "background_elements": ["finger piers", "warehouses", "parking lots"],
            "arrangement": "Ships are clustered along multiple finger piers...",
            "detail_description": "High-resolution satellite imagery captures a naval base..."
        },
        "objects_enrichment": {
            "Ship_001": {
                "class": "37",
                "visual_appearance": "Long, flat flight deck with 'X' markings...",
                "activity_status": "Stationary",
                "spatial_context": "Located at Center-Bottom..."
            },
            "Ship_002": {
                "class": "10",
                "visual_appearance": "Gray hull, blocky superstructure...",
                "activity_status": "Stationary",
                "spatial_context": "Center-Middle, very close to Ship_001..."
            }
        }
    }

    class_map = {"37": "大型航空母舰/两栖舰", "10": "现代化驱逐舰"}

    engine = VQATemplateEngine()
    sys_p, user_p = engine.get_prompts(raw_data, class_map)

    # 发送给 LLM 的内容预览
    print(sys_p)
    print(user_p)