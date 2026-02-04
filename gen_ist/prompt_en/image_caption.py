import random

class CaptionTemplateEngine:
    def __init__(self):
        # --- Style 1: Summary - 30 items ---
        self.style_1_instructions = [
            "Please briefly describe what scene this remote sensing image shows.",
            "Summarize the core content of this image in one sentence.",
            "What is the main target captured in this satellite photo?",
            "Please provide a minimalist interpretation based on the provided image information.",
            "Summarize the key geographical elements and facilities in the figure.",
            "Please briefly describe the overall situation of the area.",
            "What core targets are captured in this image?",
            "Please describe the geographical environment shown in the image in one word.",
            "Provide a preliminary overview description of this remote sensing scene.",
            "Please quickly identify and describe the theme of this interpretation material.",
            "What kind of military or civilian scene is shown in the figure?",
            "Briefly describe the main ground object features in the figure.",
            "Please summarize the interpretation results of this figure in concise language.",
            "Where does this aerial image show?",
            "Summarize the distribution of significant targets in the image.",
            "Please provide a macro scene description of the area.",
            "Briefly describe the types and environment of core facilities in the image.",
            "What kind of geographical information does this map mainly reflect?",
            "Please give a summary description of this remote sensing scene.",
            "At a glance, what does this image describe?",
            "Please confirm and briefly describe the core scene features in the image.",
            "What is the most significant feature of the image? Please explain briefly.",
            "Please provide a quick summary of this sea/land scene.",
            "Summarize the main activity scenes in the image.",
            "Please provide a short intelligence summary of the figure.",
            "What kind of port or base status does this image reflect?",
            "Briefly and concisely describe this remote sensing detection result.",
            "Please give a brief comment on the key targets and environment in the figure.",
            "What is the core geographical event captured by the image?",
            "Please give an interpretation outline for this image."
        ]

        # --- Style 2: Detailed Analysis - 30 items ---
        self.style_2_instructions = [
            "Please provide a deep professional interpretation of this image, combining environment and targets.",
            "From the perspective of intelligence analysis, describe the visual features and current status of the area in detail.",
            "Please write a detailed image analysis report by integrating weather, lighting, and object features.",
            "Please provide a meticulous detail description of all visible elements in the figure.",
            "Deeply interpret this remote sensing image, covering its environmental background and target attributes.",
            "As an interpretation expert, please describe every detail shown in the image in detail.",
            "Please provide a complete interpretation narrative of the scene, including the impact of weather on imaging.",
            "Please comprehensively analyze the visual features of the targets in the figure and their geographical environment.",
            "Provide an all-round detailed explanation of this remote sensing scene.",
            "Please deeply portray the ships and facilities in the figure, combining background elements.",
            "Describe all key features in the image to ensure complete information coverage.",
            "Please analyze the relationship between atmospheric transparency, target contours, and the environment.",
            "Please provide a detailed intelligence review of the current status of the base based on these interpretation elements.",
            "Describe every detail in the figure in detail, combining image texture and geometric features.",
            "Please write a deep interpretation text about the image, no less than 200 words.",
            "From the perspective of geographical context, detail the visual representation of the targets in the sea area.",
            "Please comprehensively analyze the geographical environment and facility status captured in the image.",
            "Detail the environmental conditions, facility layout, and target details in the image.",
            "Please provide a professional description document based on this high-resolution image.",
            "Please detail the shape, color, and integration of the targets in the figure with the surrounding environment.",
            "Provide a multi-dimensional interpretation of the scene, including background, facilities, and dynamic targets.",
            "Please provide a refined description of the image, focusing on the visual details of objects.",
            "Combining shooting time and weather, please restore the true face of the scene in detail.",
            "Please discuss the geometric structure of the targets in the figure and the surrounding facilities in detail.",
            "As an observer, please describe every element you see in this remote sensing image in detail.",
            "Please deeply analyze the facility composition of the area and its performance in the current environment.",
            "Please provide an expert-level detail description of the scene.",
            "Based on the provided raw materials, provide a long-form deep interpretation of the image.",
            "Please elaborate on the ground object association and its visual performance in the figure.",
            "Please provide an extremely detailed interpretation analysis of this maritime scene."
        ]

        # --- Style 3: Spatial Layout - 30 items ---
        self.style_3_instructions = [
            "Please describe the distribution of targets in the figure from the perspective of spatial arrangement and geographical layout.",
            "Describe the topological relationship of targets in the figure and their logical distribution in geographical space.",
            "Analyze the spatial geometric logic of ships and pier facilities in the area.",
            "How are the targets in the figure presented in linear or clustered arrangements? Please explain in detail.",
            "Please describe the relative position relationship between targets according to the geographical coordinate system.",
            "Focusing on spatial distribution, please analyze the arrangement order and density of objects in the figure.",
            "Please describe how the targets in the scene are distributed along the shoreline or pier.",
            "Analyze the spatial organization form of targets in the figure and their association with background elements.",
            "Please describe the geographical layout features of facilities and ships in the area.",
            "From the perspective of spatial architecture, explain the arrangement logic of each entity in the figure.",
            "What kind of geometric or array features does the distribution of targets in the figure present?",
            "Please explain the position logic of ships relative to piers and surrounding facilities in the figure in detail.",
            "Describe the topological structure of the scene, emphasizing the spatial orientation relationship between targets.",
            "Please analyze the spatial aggregation or dispersion patterns of targets in the figure.",
            "What is the spatial alignment of targets and their relative relationship with the environment?",
            "Please describe the grid or linear distribution of facilities in the figure from a geographical spatial perspective.",
            "Detail the current spatial arrangement of targets in the port/base in the figure.",
            "Analyze the orientation logic of targets in the figure and describe their spatial occupancy.",
            "Please describe the distribution density and arrangement trend of targets under the geographical reference system.",
            "Focusing on spatial layout, describe how objects are organized in the scene.",
            "Please analyze the geometric topological relationship of targets in the figure and explain their distribution hierarchy.",
            "Describe the cluster features of targets in the figure and their positions relative to geographical benchmarks.",
            "Please explain the arrangement relationship of targets in the scene in horizontal and vertical space.",
            "What kind of logical patterns does the spatial organization of objects in the figure present?",
            "Analyze and describe the spatial coupling relationship between ships and land facilities in the figure.",
            "Please interpret the spatial distribution of targets in the figure from the perspective of layout planning.",
            "Detail how the targets in the figure are arranged at the junction of water and land.",
            "Please provide a professional description of the spatial positioning and arrangement density of entities in the figure.",
            "Analyze the geographical distribution pattern of targets, especially their docking logic next to the pier.",
            "Please describe the spatial geometric order reflected in this remote sensing scene."
        ]

    def get_prompts(self, data):
        sys_pt = """### Role: Remote Sensing Image Interpretation Expert
You are an expert proficient in satellite remote sensing and aerial image analysis. Your task is to generate professional, vivid, and logically rigorous image descriptions (Image Caption) based on the provided structured data.

### Output Format Requirements:
The results must be returned in pure JSON format, without any Markdown format identifiers or redundant explanatory text. The JSON structure is as follows:
{
    "Style_1_Summary": {
        "Instruction": "Enter Style 1 question here",
        "Answer": "One-sentence summary of the core scene"
    },
    "Style_2_Detailed_Analysis": {
        "Instruction": "Enter Style 2 question here",
        "Answer": "Generate a deep description of about 200 words by merging environment, weather, and detailed description fields."
    },
    "Style_3_Spatial_Layout": {
        "Instruction": "Enter Style 3 question here",
        "Answer": "Focus on describing the arrangement logic of objects in geographical space (e.g., clustered, along the coast, orderly distribution)."
    }
}

### Content Writing Guidelines:
1. **De-structured Narrative**: Mechanical expressions such as "field shows" or "according to data" are strictly prohibited. The provided image information should be internalized into a first-person description of the observer.
2. **Professional Terminology Enhancement**: Use remote sensing interpretation terms appropriately. For example, transform "good visibility" into "high atmospheric transparency, high sharpness of ground object edges"; use words like "texture features", "geometric contours", "contrast", etc.
3. **Geographical Contextualization**: Metaphorize latitude and longitude information into geographical backgrounds (e.g., tropical waters, coastal waters) rather than reading numbers directly.
4. **Spatial Geometric Logic**: Focus on describing the topological relationship of targets. Use words like "linear arrangement", "grid distribution", "parallel to the shoreline", etc., to emphasize the relative position of objects and the geographical coordinate system.
5. **Prohibited Terms**: Any words about "JSON format" or "metadata" are strictly prohibited in the output content.
"""
        
        user_pt_template = """### Image Raw Information:
- Shooting Time: {time_of_day}
- Coordinates: {coordinates}
- Weather/Visibility: {weather}
- Spatial Arrangement: {arrangement}
- Visual Details: {detail}
- Background Elements: {background_elements}
- Ship Spatial Relationship: {spatial_context}

### Execution Requirements:
Please generate corresponding professional answers for the following three question templates:
Style 1: {s1_inst}
Style 2: {s2_inst}
Style 3: {s3_inst}
"""
        s1 = random.choice(self.style_1_instructions)
        s2 = random.choice(self.style_2_instructions)
        s3 = random.choice(self.style_3_instructions)

        user_pt = user_pt_template.format(
            time_of_day=data['time_of_day'],
            coordinates=data['coordinates'],
            weather=data['weather'],
            arrangement=data['arrangement'],
            detail=data['detail'],
            background_elements=data['background_elements'],
            spatial_context=data['spatial_context'],
            s1_inst=s1,
            s2_inst=s2,
            s3_inst=s3
        )

        return sys_pt, user_pt
