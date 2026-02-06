import json

class ConversationTemplateEngine:
    def __init__(self):
        # 1. 系统提示词：定义模型作为“数据集生成器”的任务
        self.sys_pt = """### 任务：生成卫星影像指令微调对话数据
你现在是一个多模态指令数据集生成器。你的任务是根据我提供的【影像原始要素全集】，创作一段“用户”与“卫星图像解译助手（Satellite Image Interpretation Agent）”之间的自然多模态对话。

### 对话参与者设定：
- **用户 (User)**：对影像内容感到好奇，或有特定的解译需求（如寻找目标、询问环境、确认坐标）。
- 卫星图像解译助手 (Agent)**：具备专业视觉解译能力的辅助型 AI。它能通过自然语言描述影像细节，并能随时调取精确的边界框坐标 [x_center, y_center, width, height] 来辅助说明。

### 对话生成准则：
1. **全要素利用**：对话内容应尽可能自然地使用提供的基本属性、背景环境、细节特征及目标属性，确保对话内容丰富多样，不需要使用全部信息。
2. **非固定轮次**：根据信息量自主决定对话长度，通常为 3-10 轮。
3. **自然且辅助**：Agent 的语气要像是在和用户共同查看图像。避免机械复读，要将结构化数据内化为“观察结果”。
4. **坐标调用**：当用户对特定物体感兴趣时，Agent 应在回复中包含该目标的精确坐标。
5. **绝对禁忌**：生成的对话文本中严禁出现“元数据”、“JSON”、“字段”或“根据记录显示”等技术词汇。
6. **指代明确**：严禁在对话中使用 Ship_001 等内部 ID。请使用视觉特征（如“那艘带飞行甲板的大船”）或空间方位（如“位于指状码头左侧的船”）进行指代。

### 输出格式：
必须返回纯 JSON 格式：
{
    "Conversation": [
        {"user": "...", "assistant": "..."},
        {"user": "...", "assistant": "..."}
    ]
}
"""

        # 2. 用户提示词模板：提供全量原始信息
        self.user_pt_template = """### 影像原始要素全集：

#### 1. 基本属性 (Metadata)
- 地面分辨率：{resolution}
- 中心坐标：经度 {lon}, 纬度 {lat}

#### 2. 场景上下文 (Scene Context)
- 场景类型：{scene_type}
- 成像环境：时间为{time_of_day}, 天气为{weather}
- 背景地物：{background_elements}
- 宏观排布：{arrangement}
- 视觉解译详情：{detail_description}

#### 3. 目标个体详情 (Objects Enrichment)
{objects_details}

### 执行：
请根据上述所有信息，生成一段“用户”与“解译助手”之间的自然对话。"""

    def _format_objects_info(self, objects_enrichment, class_map):
        """格式化所有目标信息，确保包含坐标、视觉特征和空间关联"""
        details = []
        # 创建 ID 到编号的映射，例如 {"Ship_001": "1号船"}
        id_to_num = {obj_id: f"{i+1}号船" for i, obj_id in enumerate(objects_enrichment.keys())}

        for obj_id, info in objects_enrichment.items():
            c_name = class_map.get(info['class'], "未知目标")
            
            # 替换 spatial_context 中的内部 ID
            spatial_info = info['spatial_context'].strip()
            sorted_ids = sorted(id_to_num.keys(), key=len, reverse=True)
            for old_id in sorted_ids:
                spatial_info = spatial_info.replace(old_id, id_to_num[old_id])

            detail = (
                f"- {id_to_num[obj_id]}（{c_name}）：\n"
                f"  - 边界框坐标：{info['position']}\n"
                f"  - 经纬度位置：{info.get('abs_coordinates')}\n"
                f"  - 视觉外观特征：{info['visual_appearance']}\n"
                f"  - 当前活动状态：{info['activity_status']}\n"
                f"  - 周边微观环境：{info['immediate_surroundings']}\n"
                f"  - 空间位置关联：{spatial_info}\n"
            )
            details.append(detail)
        return "\n".join(details)

    def get_prompts(self, full_json, class_map):
        """生成发送给在线模型（数据生成器）的 Payload"""
        meta = full_json.get("metadata", {})
        scene = full_json.get("scene_context", {})
        objs = full_json.get("objects_enrichment", {})

        objects_info = self._format_objects_info(objs, class_map)
        
        user_content = self.user_pt_template.format(
            resolution=meta.get("resolution"),
            lon=meta.get("center_coordinates", {}).get("longitude"),
            lat=meta.get("center_coordinates", {}).get("latitude"),
            scene_type=scene.get("scene_type"),
            time_of_day=scene.get("time_of_day"),
            weather=scene.get("weather_conditions"),
            background_elements=", ".join(scene.get("background_elements", [])),
            arrangement=scene.get("arrangement"),
            detail_description=scene.get("detail_description"),
            objects_details=objects_info
        )
        
        return self.sys_pt, user_content

# --- 模拟执行 ---
if __name__ == "__main__":
    # 模拟你提供的全量 JSON 数据
    full_data = {
        "metadata": {
            "imaging_time": "20240226",
            "resolution": "0.8m",
            "center_coordinates": {"longitude": 139.653192, "latitude": 35.289492}
        },
        "scene_context": {
            "scene_type": "Naval Base",
            "time_of_day": "Late morning to early afternoon",
            "weather_conditions": "Clear skies with good visibility",
            "background_elements": ["concrete finger piers", "large floating drydock", "warehouses"],
            "arrangement": "Ships are clustered along multiple finger piers...",
            "detail_description": "High-resolution satellite imagery captures a naval base with multiple warships..."
        },
        "objects_enrichment": {
            "Ship_001": {
                "class": "37",
                "position": [0.614031, 0.838754, 0.33248, 0.235909],
                "abs_coordinates": "35.289492N, 139.653192E",
                "visual_appearance": "The vessel features a long, flat flight deck with multiple white aircraft parking markings...",
                "activity_status": "Stationary/Docked",
                "immediate_surroundings": "Moored alongside a pier at the center-bottom...",
                "spatial_context": "Global Position: Center-Bottom..."
            }
        }
    }

    class_mapping = {"37": "航空母舰/两栖攻击舰"}
    
    generator = ConversationTemplateEngine()
    sys_prompt, user_prompt = generator.get_prompts(full_data, class_mapping)

    # 此时将 sys_prompt 和 user_prompt 发送给在线模型
    print(sys_prompt)
    print(user_prompt)