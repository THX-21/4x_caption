import json

class VisualGroundingTemplateEngine:
    def __init__(self):
        # System Prompt: Establish expert identity, task dimensions, and de-rigidification guidelines
        self.sys_pt = """### Role: Remote Sensing Image Visual Grounding Expert
You are an intelligence analysis officer proficient in satellite image interpretation and target positioning. Your task is to create diversified Visual Grounding instruction data based on the provided [Image Interpretation Materials].

### Task Core:
Guide the model to find and lock the coordinates of specific targets in the image through natural language descriptions.

### Creation Dimensions (Please ensure the generated instructions include the following multiple entry points):
1. **Absolute Orientation Positioning**: Use the global orientation of the image (e.g., center, bottom-left corner, northeast).
2. **Relative Topological Positioning**: Use the mutual positions between targets (e.g., located to the right of a certain ship, between two warships).
3. **Visual Feature Positioning**: Use the unique features of the target (e.g., flat flight deck, X-shaped marking, specific painting).
4. **Environmental Reference Positioning**: Use surrounding facilities as reference objects (e.g., berth near the warehouse, top of a certain finger pier).
5. **Unique Category Positioning**: When a certain type of target is unique in the scene, refer to it by category.

### Writing Guidelines:
- **Refuse Rigidity**: Fixed sentence structures are strictly prohibited. Randomly switch between interrogative sentences, imperative sentences, and intelligence analysis tones.
- **Professional Expression**: Use words like "target bounding box", "pixel coordinates", "topological association", "geometric center", etc.
- **De-identification**: Internal numbers such as Ship_001 are strictly prohibited.
- **Prohibited Terms**: Words like "metadata", "field", or "JSON" are strictly prohibited in the output content.

### Output Format:
Must return pure JSON list format, each element containing Instruction (diversified questions) and Answer (containing coordinates [x, y, w, h]), format as follows:
[
    {"Instruction": "...", "Answer": "..."},
    {"Instruction": "...", "Answer": "..."}
]
"""

        self.user_pt_template = """### Image Interpretation Materials:
- **Scene Overview**: {scene_info}
- **Target Distribution Details**:
{objects_details}

### Execution Requirements:
Please generate 5-8 groups of visual grounding instruction pairs for different dimensions based on the above materials. The questions should be rich and varied in perspective, reflecting an understanding of complex image environments."""

    def _format_objects_for_grounding(self, objects_enrichment, class_map):
        """Process object information into interpretation materials without IDs"""
        details = []
        # Create mapping from ID to number, e.g., {"Ship_001": "Ship 1"}
        id_to_num = {obj_id: f"Ship {i+1}" for i, obj_id in enumerate(objects_enrichment.keys())}
        
        for obj_id, info in objects_enrichment.items():
            c_name = class_map.get(info['class'], "Unknown Vessel")
            # Replace internal IDs in spatial_context
            spatial_info = info.get('spatial_context', 'Unknown position').strip()
            sorted_ids = sorted(id_to_num.keys(), key=len, reverse=True)
            for old_id in sorted_ids:
                spatial_info = spatial_info.replace(old_id, id_to_num[old_id])
            
            detail = (
                f"- {id_to_num[obj_id]} ({c_name}): Position association information: {spatial_info}. "
                f"Visual features: {info['visual_appearance']}. "
                f"Coordinate information: {info['position']}."
            )
            details.append(detail)
        return "\n".join(details)

    def get_prompts(self, raw_json, class_map):
        """Generate Payload to be sent to LLM"""
        scene = raw_json.get("scene_context", {})
        objs = raw_json.get("objects_enrichment", {})
        
        scene_info = f"In the {scene.get('scene_type')} scene, {scene.get('arrangement')}. Weather is {scene.get('weather_conditions')}."
        objects_details = self._format_objects_for_grounding(objs, class_map)
        
        user_content = self.user_pt_template.format(
            scene_info=scene_info,
            objects_details=objects_details
        )
        
        return self.sys_pt, user_content
