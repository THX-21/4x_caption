import random
import json

class DetectionTemplateEngine:
    def __init__(self):
        # --- 1. General Object Detection: Instructions and Answer Templates ---
        self.type_1_instructions = [
            "Please identify and frame all ship targets in the image.", "Identify the positions of all vessels in the image and provide bounding boxes.",
            "Locate every ship target in the figure and output its pixel coordinates.", "Detect all ships appearing in the image and return their location information.",
            "Find all vessels in the figure and provide [x_center, y_center, width, height] coordinates.", "Please perform target qualitative and positioning annotation for all visible ships in the figure.",
            "Annotate the precise geometric boundaries of each ship in the image.", "Detect all ships in the figure and output their corresponding object detection boxes.",
            "Please delineate all maritime transport tools and warships in this area.", "Identify and extract the geographical location coordinate information of all ship targets in the figure.",
            "Please locate all vessel targets in the current scene.", "What ships are in the image? Please indicate them with bounding boxes.",
            "Identify and mark all ship entities in docked or sailing states in the figure.", "Please provide the object detection box results for all vessels in the figure.",
            "Detect all ship targets in the figure and return their precise pixel positions.", "Extract the boundary position information list of all visible vessels in the figure.",
            "Please execute the full ship detection task in the given remote sensing image.", "Identify all vessels in the image and record them in coordinate form.",
            "Please delineate the distribution range of all ship targets in the figure.", "Find all vessels in this sea area or port image and annotate their boundaries.",
            "Please search for ships in the image and feedback all detection boxes.", "Identify every ship appearing in the figure and return its position boundary.",
            "Please locate all vessels in the figure to ensure no omissions.", "In this frame of image, please annotate the positions of all ships.",
            "Please perform target extraction and provide bounding boxes for all vessels in the figure.", "Identify and frame all surface ships detected in the image.",
            "Please confirm the coordinates of all ships in the figure and output them as required.", "Detect all vessel entities in the image and annotate their geometric contour boxes.",
            "Please locate all vessels in the figure through object detection algorithms.", "Give the set of coordinates for all ship targets detected in the figure."
        ]
        
        self.type_1_answer_prefixes = [
            "The bounding boxes of the ship targets detected in the image are: {data}.", "The identified ship position information is as follows: {data}.",
            "All vessels have been located in the figure, and the coordinate list is: {data}.", "The detected ship geometric boundaries are as follows: {data}.",
            "The pixel coordinate information of all visible vessels in the figure is: {data}.", "The interpreted ship distribution positions are as follows: {data}.",
            "According to the image analysis, the object boxes of all ships are: {data}.", "The set of all detected ship position coordinates: {data}.",
            "The coordinates of maritime targets identified in the image are as follows: {data}.", "The positioning results for all vessel entities are: {data}."
        ]

        # --- 2. Class-Specific Detection: Instructions and Answer Templates ---
        self.type_2_instructions = [
            "Please locate all {class_name} in the figure.", "Identify and frame all {class_name} targets in the image.",
            "Where are the {class_name} in the image? Please give their precise coordinates.", "Please detect all target entities belonging to the {class_name} class in the figure.",
            "Find the {class_name} in the figure and annotate their object detection boxes.", "Please locate all ship positions classified as {class_name}.",
            "How many {class_name} are in the figure? Please give their coordinates in the image respectively.", "Identify {class_name} in the image and extract their [x_center, y_center, width, height] information.",
            "Please give the precise bounding box sequence of all {class_name} in the figure.", "Detect and annotate every vessel belonging to the {class_name} model in the figure.",
            "Search for {class_name} in the figure and return their positions in the image coordinate system.", "Please accurately identify all {class_name} in the complex port background.",
            "Mark all {class_name} in the figure, excluding other interference targets.", "Which targets in the image belong to {class_name}? Please give their specific coordinate boxes.",
            "Please perform special target detection and positioning for {class_name} in the figure.", "Please filter and frame all {class_name} from the figure.",
            "Locate visible {class_name} in the image and output their position parameters.", "What are the detection boxes for all {class_name} in the figure?",
            "Please identify all {class_name} in the figure and indicate their range.", "In the given scene, please find all {class_name} and annotate them.",
            "Please extract the target coordinates of the specific category {class_name} in the figure.", "Identify all {class_name} in the figure to ensure accurate classification and positioning.",
            "Where are {class_name} in the figure? Please give their bounding boxes.", "Please search for and identify all {class_name} in the remote sensing image.",
            "Annotate the geographical coordinate information of targets belonging to the {class_name} category in the image.", "Please frame the {class_name} found in the figure.",
            "Find all targets matching {class_name} features and give coordinates.", "Please give the specific position distribution of {class_name} detected in the figure.",
            "Identify and feedback the bounding box list of all {class_name} targets in the figure.", "Please complete the positioning detection task for {class_name} targets."
        ]

        self.type_2_answer_prefixes = [
            "All {class_name} have been located in the image, and the coordinate list is: {data}.", "The identified {class_name} target positions in the figure are as follows: {data}.",
            "The following are all detection box coordinates belonging to the {class_name} category: {data}.", "Entities of {class_name} detected in the image, position information: {data}.",
            "According to the interpretation results, the specific coordinates of {class_name} in the figure are: {data}.", "The identified {class_name} target bounding box set is as follows: {data}.",
            "All {class_name} in the image have been annotated, coordinate information: {data}.", "After positioning, the geographical spatial coordinates of {class_name} in the figure are: {data}.",
            "The bounding boxes of {class_name} distribution in the figure are as follows: {data}.", "The pixel positioning information of all {class_name} in the figure has been extracted: {data}."
        ]

        # --- 3. Coordinate + Class Detection: Instructions and Answer Templates ---
        self.type_3_instructions = [
            "Please detect all ships in the figure and specify the specific category and coordinates of each ship.", "Identify all targets in the image and output the format as '[Category]: [Coordinates]'.",
            "Please perform classification detection for ships in the figure and give corresponding bounding boxes.", "Identify and annotate all ship targets and their corresponding model category information in the figure.",
            "Please give the category labels and precise positioning information of all vessels in the figure.", "Detect ships in the figure and distinguish their specific models and categories.",
            "Identify maritime targets in the image and provide the classification result and position box for each target.", "Please frame the ships in the figure and indicate what type of vessels they are respectively.",
            "Identify all ships in the image and return the category and detection box of each target.", "Extract the category labels and geographical spatial position information of all vessels in the figure.",
            "Please perform classification annotation and bounding box extraction for visible targets in the figure.", "Detect and identify ships in the image and output their specific ship types and position coordinate pairs.",
            "Please provide the detection results of all ships in the figure, including category names and position box data.", "Identify targets in the figure and list all findings in the form of 'Category: Coordinates'.",
            "Perform full-element detection for ships in the image, while annotating category attributes and coordinates.", "Please identify and locate all ships in the figure, requiring a one-to-one correspondence between category and coordinates.",
            "Perform classification positioning for targets in the image and give the type and framed position of each target.", "Please perform multi-target detection in the image and output the category and position coordinates of each ship.",
            "Identify and frame all ships in the figure and annotate the ship types they belong to.", "Please give the detection box and its corresponding classification name for each target in the figure.",
            "Detect all targets in the figure and output the category name combined with coordinate information.", "Please identify all ship entities in the figure and feedback their specific models and bounding boxes.",
            "What kinds of ships are in the image? Please mark their categories and position coordinates respectively.", "Please perform category annotation and position framing for each ship target detected in the figure.",
            "Identify and extract the type labels and pixel positioning boxes of all ships in the figure.", "Please output the interpretation results of each ship target in the figure, including its category and bounding box.",
            "Perform target recognition for this image, please list the types and coordinates of all ships.", "Please execute the classification detection task in the figure and annotate the model and geographical position of ships.",
            "Detect vessels in the image and return a list composed of 'Category-Coordinates'.", "Please perform a joint operation of attribute classification and spatial positioning for all ship targets in the figure."
        ]

        self.type_3_answer_prefixes = [
            "The image interpretation results (category and position) are as follows: {data}.", "The classification and coordinate information of each ship target in the figure are summarized as follows: {data}.",
            "The specific models of the identified targets and their corresponding coordinate boxes are: {data}.", "The ship entities detected in the image and their category mapping relationships are: {data}.",
            "The following are the interpretation details (category and coordinates) of all ships in the figure: {data}.", "According to the comprehensive detection, the types and position information of each target in the figure are as follows: {data}.",
            "The classification annotation and geometric boundary information of visible ships in the image: {data}.", "The category features and coordinate positions of ships in the figure have been extracted: {data}.",
            "The model identification and bounding box positioning results of each target in the figure: {data}.", "The classification detection results for all targets are as follows (Category: Coordinates): {data}."
        ]

    def generate_data(self, objects_enrichment, class_map):
        results = []
        all_objects = []
        class_groups = {}
        
        for obj_id, info in objects_enrichment.items():
            c_id = info.get("class")
            c_name = class_map.get(c_id, f"Unknown Target({c_id})")
            bbox = info.get("position")
            obj_item = {"name": c_name, "bbox": bbox}
            all_objects.append(obj_item)
            if c_name not in class_groups:
                class_groups[c_name] = []
            class_groups[c_name].append(bbox)

        # --- Generate Type 1 (General Detection) ---
        t1_inst = random.choice(self.type_1_instructions)
        t1_data_str = ", ".join([str(o["bbox"]) for o in all_objects])
        t1_ans = random.choice(self.type_1_answer_prefixes).format(data=t1_data_str)
        results.append({"type": "General_Detection", "instruction": t1_inst, "answer": t1_ans})

        # --- Generate Type 2 (Class-Specific Detection) ---
        if class_groups:
            target_class = random.choice(list(class_groups.keys()))
            t2_inst = random.choice(self.type_2_instructions).format(class_name=target_class)
            t2_data_str = ", ".join([str(bbox) for bbox in class_groups[target_class]])
            t2_ans = random.choice(self.type_2_answer_prefixes).format(class_name=target_class, data=t2_data_str)
            results.append({"type": "Class_Specific", "instruction": t2_inst, "answer": t2_ans})

        # --- Generate Type 3 (Comprehensive Detection: Category + Coordinates) ---
        t3_inst = random.choice(self.type_3_instructions)
        t3_data_str = "; ".join([f"{o['name']}: {o['bbox']}" for o in all_objects])
        t3_ans = random.choice(self.type_3_answer_prefixes).format(data=t3_data_str)
        results.append({"type": "Comprehensive_Detection", "instruction": t3_inst, "answer": t3_ans})

        return results
