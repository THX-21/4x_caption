import json

class VQATemplateEngine:
    def __init__(self):
        # System Prompt: Set expert role, task ratio, and context constraints
        self.sys_pt = """### Role: Remote Sensing Image Interpretation Expert
You are an expert proficient in satellite remote sensing and aerial image intelligence analysis. Your task is to build a set of high-quality instruction fine-tuning Visual Question Answering (VQA) pairs based on the provided structured image data.

### Task Core Requirements:
1. **Task Ratio (Important)**:
   - **Specific Information Q&A (approx. 75%)**: Questions about imaging time, resolution, weather, visual features of specific targets (e.g., color, markings, masts), background buildings, precise spatial relationships (e.g., top-left, distance), etc.
   - **Summary and Reasoning Q&A (approx. 25%)**: Combine multiple fields to infer scene nature, base activity, impact of environment on observation, or summary of overall layout logic, etc.

2. **Writing Guidelines**:
   - **De-structured Narrative**: Words like "according to data" or "field shows" are strictly prohibited. Q&A should be like looking at a real photo.
   - **Professional Terminology**: Use words like "ground object contour", "geometric structure", "atmospheric transparency", "berth utilization", "topological association", etc., appropriately.
   - **De-identification**: Using IDs like Ship_001 in questions is strictly prohibited; targets must be referred to by spatial position or visual features.
   - **Prohibited Terms**: Words like "metadata" or "JSON" are strictly prohibited in the output content.

### Output Format Requirements:
Must return in pure JSON list format, each element containing Instruction and Answer. Format as follows:
[
    {"Instruction": "Specific information question...", "Answer": "Fact-based answer..."},
    {"Instruction": "Comprehensive reasoning question...", "Answer": "Logic-based analysis..."}
]
"""

        self.user_pt_template = """### Image Raw Materials:
- **Basic Attributes**: Imaging time is {imaging_time}, ground resolution is {resolution}, center coordinates {coordinates}.
- **Environmental Background**: Scene is {scene_type}. Weather at the time: {weather}.
- **Background Elements**: Surrounding exists {background_elements}.
- **Macro Layout**: {arrangement}
- **Detailed Interpretation Description**: {detail_description}
- **Target Individual Features and Spatial Logic**:
{objects_info_block}

### Execution Task:
Please generate a set of (10-12 in total) professional Q&A pairs based on the above materials, ensuring coverage of specific detail inquiries and a moderate amount of comprehensive reasoning analysis.
"""

    def _format_objects_info(self, objects_enrichment, class_map):
        """Process object information into a more readable text block, hide internal IDs, convert category names"""
        info_blocks = []
        # Create mapping from ID to number, e.g., {"Ship_001": "Ship 1"}
        id_to_num = {obj_id: f"Ship {i+1}" for i, obj_id in enumerate(objects_enrichment.keys())}
        
        for obj_id, info in objects_enrichment.items():
            c_name = class_map.get(info['class'], "Unknown Vessel")
            # Replace internal IDs in spatial_context
            spatial_info = info['spatial_context'].strip()
            # Sort IDs by length in descending order to prevent Ship_0011 from incorrectly matching Ship_001
            sorted_ids = sorted(id_to_num.keys(), key=len, reverse=True)
            for old_id in sorted_ids:
                spatial_info = spatial_info.replace(old_id, id_to_num[old_id])
            
            desc = (
                f"- {id_to_num[obj_id]} ({c_name}): Appearance features are \"{info['visual_appearance']}\"; "
                f"Current status is \"{info['activity_status']}\"; "
                f"Position association information is: {spatial_info}"
            )
            info_blocks.append(desc)
        return "\n".join(info_blocks)

    def get_prompts(self, raw_json, class_map):
        """
        Generate final system and user prompts to be sent to LLM
        """
        meta = raw_json.get("metadata", {})
        scene = raw_json.get("scene_context", {})
        objs = raw_json.get("objects_enrichment", {})

        # Pre-process fields to avoid the word "metadata"
        objects_info_block = self._format_objects_info(objs, class_map)
        
        user_content = self.user_pt_template.format(
            imaging_time=meta.get("imaging_time"),
            resolution=meta.get("resolution"),
            coordinates=f"{meta.get('center_coordinates', {}).get('latitude')}N, {meta.get('center_coordinates', {}).get('longitude')}E",
            scene_type=scene.get("scene_type"),
            weather=scene.get("weather_conditions"),
            background_elements=", ".join(scene.get("background_elements", [])),
            arrangement=scene.get("arrangement"),
            detail_description=scene.get("detail_description"),
            objects_info_block=objects_info_block
        )

        return self.sys_pt, user_content
