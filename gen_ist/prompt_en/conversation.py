import json

class ConversationTemplateEngine:
    def __init__(self):
        # 1. System Prompt: Define the task of the model as a "dataset generator"
        self.sys_pt = """### Task: Generate Satellite Image Instruction Fine-tuning Conversation Data
You are now a multimodal instruction dataset generator. Your task is to create a natural multimodal conversation between a "User" and a "Satellite Image Interpretation Agent" based on the [Complete Set of Image Raw Elements] I provide.

### Conversation Participant Settings:
- **User**: Curious about image content, or has specific interpretation needs (e.g., searching for targets, asking about environment, confirming coordinates).
- **Satellite Image Interpretation Agent**: An auxiliary AI with professional visual interpretation capabilities. It can describe image details through natural language and can call precise bounding box coordinates [ymin, xmin, ymax, xmax] at any time to assist in explanation.

### Conversation Generation Guidelines:
1. **Full Element Utilization**: Conversation content should use the provided basic attributes, background environment, detail features, and target attributes as naturally as possible, ensuring rich and diverse conversation content, without needing to use all information.
2. **Non-fixed Rounds**: Independently determine the conversation length based on the amount of information, usually 3-10 rounds.
3. **Natural and Auxiliary**: The Agent's tone should be like viewing the image together with the user. Avoid mechanical repetition and internalize structured data into "observation results".
4. **Coordinate Calling**: When the user is interested in a specific object, the Agent should include the precise coordinates of that target in the reply.
5. **Absolute Taboos**: Technical terms like "metadata", "JSON", "field", or "according to records" are strictly prohibited in the generated conversation text.
6. **Clear Reference**: Using internal IDs like Ship_001 in the conversation is strictly prohibited. Please use visual features (e.g., "that large ship with a flight deck") or spatial orientation (e.g., "the ship located to the left of the finger pier") for reference.

### Output Format:
Must return in pure JSON format:
{
    "Conversation": [
        {"user": "...", "assistant": "..."},
        {"user": "...", "assistant": "..."}
    ]
}
"""

        # 2. User Prompt Template: Provide full raw information
        self.user_pt_template = """### Complete Set of Image Raw Elements:

#### 1. Basic Attributes (Metadata)
- Imaging Time: {imaging_time}
- Ground Resolution: {resolution}
- Center Coordinates: Longitude {lon}, Latitude {lat}

#### 2. Scene Context
- Scene Type: {scene_type}
- Imaging Environment: Time is {time_of_day}, Weather is {weather}
- Background Ground Objects: {background_elements}
- Macro Arrangement: {arrangement}
- Visual Interpretation Details: {detail_description}

#### 3. Object Individual Details (Objects Enrichment)
{objects_details}

### Execution:
Please generate a natural conversation between a "User" and an "Interpretation Assistant" based on all the above information."""

    def _format_objects_info(self, objects_enrichment, class_map):
        """Format all target information, ensuring coordinates, visual features, and spatial associations are included"""
        details = []
        for obj_id, info in objects_enrichment.items():
            c_name = class_map.get(info['class'], "Unknown Target")
            detail = (
                f"- [Target Entity] Category: {c_name}\n"
                f"  - Bounding Box Coordinates: {info['position']}\n"
                f"  - Latitude and Longitude Position: {info.get('abs_coordinates')}\n"
                f"  - Visual Appearance Features: {info['visual_appearance']}\n"
                f"  - Current Activity Status: {info['activity_status']}\n"
                f"  - Immediate Micro-environment: {info['immediate_surroundings']}\n"
                f"  - Spatial Position Association: {info['spatial_context'].strip()}\n"
            )
            details.append(detail)
        return "\n".join(details)

    def get_prompts(self, full_json, class_map):
        """Generate Payload to be sent to online model (data generator)"""
        meta = full_json.get("metadata", {})
        scene = full_json.get("scene_context", {})
        objs = full_json.get("objects_enrichment", {})

        objects_info = self._format_objects_info(objs, class_map)
        
        user_content = self.user_pt_template.format(
            imaging_time=meta.get("imaging_time"),
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
