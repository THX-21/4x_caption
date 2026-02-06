import random

class CaptionTemplateEngine:
    def __init__(self):
        # --- Style 1: 简明摘要 (Summary) - 30条 ---
        self.style_1_instructions = [
            "请简要说明这幅遥感图像展示了什么场景。",
            "用一句话概括这幅影像的核心内容。",
            "这张卫星照片拍摄的主要目标是什么？",
            "请根据提供的影像信息进行极简解译。",
            "概括图中的关键地理要素与设施。",
            "请简述该区域的整体态势。",
            "这幅图中捕捉到了哪些核心目标？",
            "请一语道破该影像展示的地理环境。",
            "对此遥感画面进行初步的概览描述。",
            "请快速识别并描述该解译资料的主题。",
            "图中展现了怎样的军事或民用场景？",
            "简要描述图中的主要地物特征。",
            "请用精炼的语言概括本图的解译结果。",
            "这幅航拍影像展示了什么地方？",
            "概括图像中显著的目标分布情况。",
            "请提供该区域的宏观场景描述。",
            "简述该影像中核心设施的类型与环境。",
            "这张图主要反映了什么样的地理信息？",
            "请给出该遥感场景的摘要式描述。",
            "一眼看去，这幅图像描述的是什么？",
            "请确认并简述图像中的核心场景特征。",
            "该影像最显著的特征是什么？请简要说明。",
            "请对该海域/陆地场景进行快速概括。",
            "概括该影像中的主要活动场景。",
            "请提供关于该图的简短情报摘要。",
            "这幅图反映了怎样的港口或基地现状？",
            "简明扼要地描述此遥感探测结果。",
            "请对图中的关键目标及环境进行简评。",
            "该影像捕捉到的核心地理事件是什么？",
            "请给出一句关于该图像的解译大纲。"
        ]

        # --- Style 2: 详细分析 (Detailed) - 30条 ---
        self.style_2_instructions = [
            "请结合环境与目标，对这幅影像进行深度的专业解译。",
            "从情报分析的角度，详细描述该区域的视觉特征与现状。",
            "请融合天气、光照及物体特征，撰写一份详尽的影像分析报告。",
            "请对图中所有可见要素进行细致入微的细节描述。",
            "深入解读这幅遥感影像，涵盖其环境背景与目标属性。",
            "作为解译专家，请详细描述该影像展示的每一个细节。",
            "请提供该场景的完整解译叙述，包含天气对成像的影响。",
            "请全面分析图中目标的视觉特征及其所处的地理环境。",
            "对该遥感场景进行全方位的详细解说。",
            "请结合背景元素，对图中的舰船及设施进行深度刻画。",
            "描述该影像中的所有关键特征，确保信息覆盖完整。",
            "请分析大气的透明度、目标轮廓以及环境的相互关系。",
            "请根据这些解译要素，对该基地的现状进行详尽的情报述评。",
            "结合图像纹理与几何特征，详细描述图中的每一处细节。",
            "请撰写一段关于该影像的深度解译文本，不少于200字。",
            "从地理语境出发，详述该海域目标的视觉表征。",
            "请全面解析该影像中捕捉到的地理环境与设施状态。",
            "详细说明图像中的环境条件、设施排布及目标细节。",
            "请基于该高分辨率影像，提供一份专业的描述文档。",
            "请详述图中目标的形态、颜色及周围环境的融合情况。",
            "对该场景进行多维度的解译，包括背景、设施与动态目标。",
            "请提供该影像的精细化描述，重点关注物体的视觉细节。",
            "结合拍摄时间与天气，请详细还原该场景的真实面貌。",
            "请对图中目标的几何结构与周围设施进行详细论述。",
            "作为观察者，请详细描述你在这幅遥感图中看到的每一个要素。",
            "请深入分析该区域的设施构成及其在当前环境下的表现。",
            "请提供关于该场景的专家级细节描述。",
            "根据提供的原始素材，对该影像进行长篇幅的深度解读。",
            "请详细阐述图中的地物关联及其视觉表现。",
            "请对该海事场景进行极其详尽的解译分析。"
        ]

        # --- Style 3: 空间布局 (Spatial) - 30条 ---
        self.style_3_instructions = [
            "请从空间排列和地理布局的角度描述图中目标的分布情况。",
            "描述图中目标的拓扑关系及其在地理空间内的逻辑分布。",
            "分析该区域内舰船与码头设施的空间几何逻辑。",
            "图中目标是如何呈现线性或成簇排列的？请详细说明。",
            "请根据地理坐标系，描述各目标之间的相对位置关系。",
            "侧重于空间分布，请分析图中物体的排列顺序与密度。",
            "请描述该场景中的目标是如何沿岸线或码头进行分布的。",
            "分析图中目标的空间组织形式及其与背景元素的关联。",
            "请描述该区域内设施与船只的地理布局特征。",
            "从空间架构角度，解说图中各实体的排布逻辑。",
            "图中目标的分布呈现出怎样的几何或阵列特征？",
            "请详细说明图中舰船相对于码头和周围设施的位置逻辑。",
            "描述该场景的拓扑结构，强调目标间的空间方位关系。",
            "请分析图中各目标在空间上的聚合或离散规律。",
            "目标的空间对齐方式及其与环境的相对关系是怎样的？",
            "请从地理空间视角，描述图中设施的网格化或线性分布。",
            "详细叙述图中目标在港口/基地内的空间排布现状。",
            "分析图中目标的方位逻辑，描述其空间占位情况。",
            "请描述目标在地理参考系下的分布密度与排列走势。",
            "侧重于空间布局，描述物体是如何在场景中进行组织的。",
            "请分析图中目标的几何拓扑关系，说明其分布层级。",
            "描述图中目标的集群特征及其相对于地理基准的位置。",
            "请说明场景中各目标在水平与垂直空间上的排列关系。",
            "图中物体的空间组织呈现出怎样的逻辑规律？",
            "分析并描述图中舰船与陆域设施的空间耦合关系。",
            "请从布局规划的角度，解读图中目标的空间分布。",
            "详细描述图中目标是如何在水域与陆域交界处排列的。",
            "请对图中实体的空间定位与排列密度进行专业描述。",
            "分析目标的地理分布格局，特别是其在码头旁的停泊逻辑。",
            "请描述该遥感场景中体现出的空间几何秩序。"
        ]

    def flatten_data(self, raw_data):
        """为 CaptionEngine 准备扁平化数据"""
        # 合并所有目标的 spatial_context
        spatial_contexts = []
        for obj_id, obj_data in raw_data.get("objects_enrichment", {}).items():
            if "spatial_context" in obj_data:
                spatial_contexts.append(obj_data["spatial_context"])
        
        combined_spatial_context = " ".join(spatial_contexts) if spatial_contexts else "N/A"
        
        return {
            'time_of_day': raw_data["scene_context"]["time_of_day"],
            'coordinates': f"{raw_data['metadata']['center_coordinates']['latitude']}N, {raw_data['metadata']['center_coordinates']['longitude']}E",
            'weather': raw_data["scene_context"]["weather_conditions"],
            'arrangement': raw_data["scene_context"]["arrangement"],
            'detail': raw_data["scene_context"]["detail_description"],
            'background_elements': ", ".join(raw_data["scene_context"]["background_elements"]),
            'spatial_context': combined_spatial_context
        }

    def get_prompts(self, data):
        """
        data: 传入包含 time_of_day, coordinates, weather 等字段的字典
        """
        sys_pt = """### 角色：遥感影像解译专家
你是一名精通卫星遥感与航拍影像分析的专家。你的任务是根据提供的结构化数据，生成专业、生动且逻辑严密的影像描述（Image Caption）。

### 输出格式要求：
必须以纯 JSON 格式返回结果，不得包含任何 Markdown 格式标识或多余的解释文字。JSON 结构如下：
{
    "Style_1_Summary": {
        "Instruction": "这里填入Style 1的提问",
        "Answer": "一句话概括核心场景"
    },
    "Style_2_Detailed_Analysis": {
        "Instruction": "这里填入Style 2的提问",
        "Answer": "融合环境、天气及详细描述字段，生成 200 字左右的深度描述。"
    },
    "Style_3_Spatial_Layout": {
        "Instruction": "这里填入Style 3的提问",
        "Answer": "侧重于描述物体在地理空间内的排列逻辑（如成簇、沿岸、有序分布）。"
    }
}

### 内容撰写指南：
1. **去结构化叙述**：严禁出现“字段显示”、“根据数据”等机械表述。需将提供的影像信息内化为观察者的第一视角描述。
2. **专业术语增强**：恰当使用遥感解译术语。如将“能见度好”转化为“大气透明度高，地物边缘锐度高”；使用“纹理特征”、“几何轮廓”、“对比度”等词汇。
3. **地理语境化**：将经纬度信息隐喻为地理背景（如：热带海域、近岸水域），而非直接朗读数字。
4. **空间几何逻辑**：侧重描述目标的拓扑关系。使用“线性排列”、“网格化分布”、“与岸线保持平行”等词汇，强调物体与地理坐标系的相对位置。
5. **严禁术语**：严禁在输出内容中出现任何关于“JSON格式”或“元数据”的字眼。
"""
        
        user_pt_template = """### 影像原始信息：
- 拍摄时间：{time_of_day}
- 经纬度：{coordinates}
- 天气/能见度：{weather}
- 空间排列：{arrangement}
- 视觉细节：{detail}
- 背景元素：{background_elements}
- 舰船空间位置关系：{spatial_context}

### 执行要求：
请分别针对以下三个提问模板生成对应的专业回答：
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
if __name__ == "__main__":
    # --- 使用示例 ---
    engine = CaptionTemplateEngine()
    # 模拟数据
    sample_data = {
        "time_of_day": "Late morning",
        "coordinates": "35.28N, 139.65E",
        "weather": "Clear skies",
        "arrangement": "Clustered along piers",
        "detail": "Warships moored at concrete piers...",
        "background_elements": "Warehouses, parking lots",
        "spatial_context": "Ship_001 is docked at the main pier..."
    }

    sys_p, user_p = engine.get_prompts(sample_data)
    print(sys_p)
    print(user_p)
    # 接下来将 sys_p 和 user_p 发送给 LLM 即可