import random
import json

class DetectionTemplateEngine:
    def __init__(self):
        # --- 1. 全目标检测：指令与回答模板 ---
        self.type_1_instructions = [
            "请在图像中识别并框选出所有的舰船目标。", "识别影像中所有船只的位置并给出边界框。",
            "定位图中每一处舰船目标，输出其像素坐标。", "检测影像中所有出现的舰船，并返回它们的位置信息。",
            "找出图中的所有船只并给出 [x_center, y_center, width, height] 坐标。", "请对图中可见的全部舰船进行目标定性与定位标注。",
            "标注出影像中每一艘船的精确几何边界。", "检测图中所有舰船，输出其对应的目标检测框。",
            "请圈定本区域内所有的海上交通工具及舰艇。", "识别并提取图中所有舰船目标的地理位置坐标信息。",
            "请在当前场景中定位所有的船只目标。", "图像中有哪些舰船？请通过边界框标示出来。",
            "识别并标记出图中所有处于停泊或航行状态的舰船实体。", "请给出图中所有船只的目标检测框结果。",
            "探测图中的所有舰船目标并返回其精确的像素位置。", "提取图中所有可见船只的边界位置信息列表。",
            "请在给定的遥感影像中执行全量舰船检测任务。", "识别图像中的所有船只，并以坐标形式进行记录。",
            "请划定图中所有舰船目标的分布范围。", "找出该海域或港口影像中所有的船只，并标注其边界。",
            "请对影像中的舰船进行搜索，并反馈所有的检测框。", "识别图中出现的每一艘舰船并返回其位置边界。",
            "请定位图中的所有船舶，确保无遗漏。", "在这一帧影像中，请标注出所有舰船的位置。",
            "请执行目标提取，给出图中所有船只的边界框。", "识别并框选影像中检测到的所有水面舰船。",
            "请确认图中所有舰船的坐标，并按要求输出。", "检测影像中所有的船只实体，标注其几何轮廓框。",
            "请通过目标检测算法定位图中所有的船只。", "给出图中探测到的全部舰船目标的坐标集合。"
        ]
        
        self.type_1_answer_prefixes = [
            "影像中检测到的舰船目标边界框为：{data}。", "识别出的舰船位置信息如下：{data}。",
            "已在图中定位到所有船只，坐标列表为：{data}。", "探测到的舰船几何边界如下：{data}。",
            "图中所有可见船只的像素坐标信息为：{data}。", "解译出的舰船分布位置如下：{data}。",
            "根据影像分析，所有舰船的目标框为：{data}。", "检测到的全部舰船位置坐标集合：{data}。",
            "影像中识别出的海上目标坐标如下：{data}。", "所有船只实体的定位结果为：{data}。"
        ]

        # --- 2. 特定类别检测：指令与回答模板 ---
        self.type_2_instructions = [
            "请在图中定位所有的{class_name}。", "识别并框选出影像中所有的{class_name}目标。",
            "图像中的{class_name}位于哪里？请给出它们的精确坐标。", "请检测图中所有属于{class_name}类的目标实体。",
            "找出图中的{class_name}，并标注其目标检测框。", "请定位出所有分类标注为{class_name}的舰船位置。",
            "图中有多少个{class_name}？请分别给出它们在影像中的坐标。", "识别影像中的{class_name}并提取其[x_center, y_center, width, height]信息。",
            "请给出图中所有{class_name}的精确边界框序列。", "检测并标注出图里每一艘属于{class_name}型号的船只。",
            "搜索图中的{class_name}，并返回其在图像坐标系下的位置。", "请在复杂的港口背景中精准识别出所有的{class_name}。",
            "标记出图中所有的{class_name}，排除其他干扰目标。", "图像中哪些目标属于{class_name}？请给出其具体的坐标框。",
            "请对图中的{class_name}进行专项目标检测与定位。", "请从图中筛选并框选出所有的{class_name}。",
            "定位影像中可见的{class_name}，输出其位置参数。", "图中所有{class_name}的检测框是什么？",
            "请识别出图中所有的{class_name}并标明其范围。", "在给定的场景中，请找出所有的{class_name}并标注。",
            "请提取图中{class_name}这一特定类别的目标坐标。", "识别图中所有的{class_name}，确保分类与定位准确。",
            "图中哪里有{class_name}？请给出它们的边界框。", "请在遥感图像中搜索并识别出所有的{class_name}。",
            "标注影像中属于{class_name}类别目标的地理坐标信息。", "请对图中发现的{class_name}进行框选。",
            "找出所有符合{class_name}特征的目标并给出坐标。", "请给出图中检测到的{class_name}的具体位置分布。",
            "识别并反馈图中所有{class_name}目标的边界框列表。", "请完成对{class_name}目标的定位检测任务。"
        ]

        self.type_2_answer_prefixes = [
            "已在影像中定位到所有的{class_name}，坐标列表为：{data}。", "图中识别出的{class_name}目标位置如下：{data}。",
            "以下是所有属于{class_name}类别的检测框坐标：{data}。", "探测到影像中的{class_name}实体，位置信息为：{data}。",
            "根据解译结果，图中{class_name}的具体坐标为：{data}。", "识别到的{class_name}目标边界框集合如下：{data}。",
            "影像中所有的{class_name}已标注，坐标信息：{data}。", "经定位，图中{class_name}的地理空间坐标为：{data}。",
            "图中{class_name}分布的边界框如下：{data}。", "已提取出图中所有{class_name}的像素定位信息：{data}。"
        ]

        # --- 3. 坐标+类别检测：指令与回答模板 ---
        self.type_3_instructions = [
            "请检测图中的所有舰船，并注明每艘船的具体类别与坐标。", "识别影像中的所有目标，输出格式为“[类别]: [坐标]”。",
            "请对图中的舰船进行分类检测，并给出对应的边界框。", "识别并标注图中所有的舰船目标及其对应的型号类别信息。",
            "请给出图中所有船只的类别标签及精确的定位信息。", "检测图中的舰船，并区分出它们的具体型号和类别。",
            "识别图像中的海上目标，并提供每个目标的分类结果和位置框。", "请框选出图中的舰船，并指明它们分别是哪种类型的船只。",
            "对影像中的所有舰船进行识别，返回各目标的类别及其检测框。", "提取图中所有船只的类别标签与地理空间位置信息。",
            "请对图中可见的目标进行分类标注与边界框提取工作。", "检测并识别影像中的舰船，输出其具体的船型和位置坐标对。",
            "请提供图中所有舰船的检测结果，包含类别名称与位置框数据。", "识别图中的目标，并以‘类别：坐标’的形式列出所有发现。",
            "对影像中的舰船进行全要素检测，同时标注出类别属性与坐标。", "请识别并定位图中的所有舰船，要求类别与坐标一一对应。",
            "对图像中的目标进行分类定位，给出各目标的类型与框选位置。", "请在影像中进行多目标检测，输出各舰船的类别及其位置坐标。",
            "识别并框选图中所有的舰船，并标注它们所属的船种。", "请给出图中每一个目标的检测框及其对应的分类名称。",
            "检测图中的所有目标，并将类别名与坐标信息结合输出。", "请识别图中所有的舰船实体，并反馈它们的具体型号与边界框。",
            "图像中有哪些种类的船？请分别标出它们的类别和位置坐标。", "请对图中检测到的每个舰船目标进行类别标注和位置框定。",
            "识别并提取图中所有舰船的类型标签及像素定位框。", "请输出图中各舰船目标的解译结果，包含其类别和边界框。",
            "对该影像进行目标识别，请列出所有舰船的种类及其坐标。", "请在图中执行分类检测任务，标注出舰船的型号与地理位置。",
            "检测影像中的船只，并返回由‘类别-坐标’组成的列表。", "请对图中所有舰船目标进行属性分类与空间定位的联合作业。"
        ]

        self.type_3_answer_prefixes = [
            "影像解译结果（类别与位置）如下：{data}。", "图中各舰船目标的分类及坐标信息汇总如下：{data}。",
            "识别出的目标具体型号及其对应的坐标框为：{data}。", "检测到影像中的舰船实体及其类别映射关系为：{data}。",
            "以下是图中所有舰船的解译明细（类别及坐标）：{data}。", "根据综合检测，图中各目标的种类与位置信息如下：{data}。",
            "影像中可见舰船的分类标注与几何边界信息：{data}。", "已提取图中舰船的类别特征与坐标位置：{data}。",
            "图中各目标的型号辨识及边界框定位结果：{data}。", "全量目标的分类检测结果如下（类别：坐标）：{data}。"
        ]

    def generate_data(self, objects_enrichment, class_map):
        results = []
        all_objects = []
        class_groups = {}
        
        for obj_id, info in objects_enrichment.items():
            c_id = info.get("class")
            c_name = class_map.get(c_id, f"未知目标({c_id})")
            bbox = info.get("position")
            obj_item = {"name": c_name, "bbox": bbox}
            all_objects.append(obj_item)
            if c_name not in class_groups:
                class_groups[c_name] = []
            class_groups[c_name].append(bbox)

        # --- 生成 Type 1 (全目标检测) ---
        t1_inst = random.choice(self.type_1_instructions)
        t1_data_str = ", ".join([str(o["bbox"]) for o in all_objects])
        t1_ans = random.choice(self.type_1_answer_prefixes).format(data=t1_data_str)
        results.append({"type": "General_Detection", "instruction": t1_inst, "answer": t1_ans})

        # --- 生成 Type 2 (特定类别检测) ---
        if class_groups:
            target_class = random.choice(list(class_groups.keys()))
            t2_inst = random.choice(self.type_2_instructions).format(class_name=target_class)
            t2_data_str = ", ".join([str(bbox) for bbox in class_groups[target_class]])
            t2_ans = random.choice(self.type_2_answer_prefixes).format(class_name=target_class, data=t2_data_str)
            results.append({"type": "Class_Specific", "instruction": t2_inst, "answer": t2_ans})

        # --- 生成 Type 3 (综合检测：类别+坐标) ---
        t3_inst = random.choice(self.type_3_instructions)
        t3_data_str = "; ".join([f"{o['name']}: {o['bbox']}" for o in all_objects])
        t3_ans = random.choice(self.type_3_answer_prefixes).format(data=t3_data_str)
        results.append({"type": "Comprehensive_Detection", "instruction": t3_inst, "answer": t3_ans})

        return results

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟输入数据
    my_class_map = {"37": "航空母舰", "10": "驱逐舰", "9": "护卫舰"}
    sample_objects = {
        "Ship_001": {"class": "37", "position": [0.61, 0.83, 0.33, 0.23]},
        "Ship_002": {"class": "10", "position": [0.49, 0.49, 0.12, 0.17]},
        "Ship_003": {"class": "10", "position": [0.47, 0.49, 0.11, 0.16]}
    }

    engine = DetectionTemplateEngine()
    it_data = engine.generate_data(sample_objects, my_class_map)

    for i, data in enumerate(it_data):
        print(f"任务类型 {i+1}:")
        print(f"Q: {data['instruction']}")
        print(f"A: {data['answer']}\n")