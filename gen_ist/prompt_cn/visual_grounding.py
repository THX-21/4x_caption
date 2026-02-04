import json

class VisualGroundingTemplateEngine:
    def __init__(self):
        # 系统提示词：确立专家身份、任务维度及去死板化准则
        self.sys_pt = """### 角色：遥感影像视觉定位专家
你是一名精通卫星影像解译与目标定位的情报分析官。你的任务是基于提供的【影像解译素材】，创作多样化的视觉定位（Visual Grounding）指令数据。

### 任务核心：
通过自然语言描述引导模型在图像中寻找并锁定特定目标的坐标。

### 创作维度（请确保生成的指令包含以下多种切入点）：
1. **绝对方位定位**：利用影像的全局方位（如：中心、左下角、东北部）。
2. **相对拓扑定位**：利用目标间的相互位置（如：位于某船只右侧、两舰之间）。
3. **视觉特征定位**：利用目标的独有特征（如：平直飞行甲板、X型标识、特定涂装）。
4. **环境参照定位**：利用周边设施作为参照物（如：靠近仓库的泊位、某指状码头顶端）。
5. **唯一类别定位**：当场景中某类目标唯一时，通过类别指代。

### 撰写准则：
- **拒绝死板**：严禁使用固定句式。随机切换疑问句、祈使句及情报分析口吻。
- **专业表达**：使用“目标边界框”、“像素坐标”、“拓扑关联”、“几何中心”等词汇。
- **去标识化**：严禁出现 Ship_001 等内部编号。
- **严禁术语**：输出内容中严禁出现“元数据”、“字段”或“JSON”字样。

### 输出格式：
必须返回纯 JSON 列表格式，每个元素包含 Instruction（多样化的提问）和 Answer（包含坐标 [x, y, w, h]），格式如下：
[
    {{"Instruction": "...", "Answer": "..."}},
    {{"Instruction": "...", "Answer": "..."}}
]
"""

        self.user_pt_template = """### 影像解译素材：
- **场景概况**：{scene_info}
- **目标分布详情**：
{objects_details}

### 执行要求：
请根据上述素材，为本场景生成 5-8 组不同维度的视觉定位指令对。要求提问的角度丰富多变，能够体现出对复杂影像环境的理解。"""

    def _format_objects_for_grounding(self, objects_enrichment, class_map):
        """将对象信息处理为不含 ID 的解译素材"""
        details = []
        # 创建 ID 到编号的映射，例如 {"Ship_001": "1号船"}
        id_to_num = {obj_id: f"{i+1}号船" for i, obj_id in enumerate(objects_enrichment.keys())}
        
        for obj_id, info in objects_enrichment.items():
            c_name = class_map.get(info['class'], "未知船只")
            # 替换 spatial_context 中的内部 ID
            spatial_info = info.get('spatial_context', '未知位置').strip()
            sorted_ids = sorted(id_to_num.keys(), key=len, reverse=True)
            for old_id in sorted_ids:
                spatial_info = spatial_info.replace(old_id, id_to_num[old_id])
            
            detail = (
                f"- {id_to_num[obj_id]}（{c_name}）：位置关联信息为：{spatial_info}。 "
                f"视觉特征：{info['visual_appearance']}。 "
                f"坐标信息：{info['position']}。"
            )
            details.append(detail)
        return "\n".join(details)

    def get_prompts(self, raw_json, class_map):
        """生成发送给 LLM 的 Payload"""
        scene = raw_json.get("scene_context", {})
        objs = raw_json.get("objects_enrichment", {})
        
        scene_info = f"在{scene.get('scene_type')}场景下，{scene.get('arrangement')}。天气为{scene.get('weather_conditions')}。"
        objects_details = self._format_objects_for_grounding(objs, class_map)
        
        user_content = self.user_pt_template.format(
            scene_info=scene_info,
            objects_details=objects_details
        )
        
        return self.sys_pt, user_content

# --- 模拟执行流程 ---
if __name__ == "__main__":
    raw_data = {
        "scene_context": {
            "scene_type": "Naval Base",
            "arrangement": "Ships are clustered along multiple finger piers...",
            "weather_conditions": "Clear skies with good visibility"
        },
        "objects_enrichment": {
            "Ship_001": {
                "class": "37",
                "position": [0.61, 0.83, 0.33, 0.23],
                "visual_appearance": "Long, flat flight deck with white 'X' patterns...",
                "spatial_context": "Global Position: Center-Bottom..."
            },
            "Ship_002": {
                "class": "37",
                "position": [0.61, 0.83, 0.33, 0.23],
                "visual_appearance": "Long, flat flight deck with white 'X' patterns...",
                "spatial_context": "Global Position: Center-Bottom, Very Close to Ship_001..."
            }
        }
    }
    class_map = {"37": "航空母舰"}

    engine = VisualGroundingTemplateEngine()
    sys_p, user_p = engine.get_prompts(raw_data, class_map)
    print(sys_p)
    print(user_p)