import os, json, re
import rasterio

def extract_normalized_info(tif_path, json_path):
    # 1. 获取地理变换参数
    with rasterio.open(tif_path) as ds:
        gt = ds.transform
        width, height = ds.width, ds.height
        # rasterio transform: (a, b, c, d, e, f) 对应 GDAL: (c, a, b, f, d, e)
        # gt[2] 是左上角 x, gt[0] 是 x 分辨率, gt[1] 是旋转
        # gt[5] 是左上角 y, gt[3] 是旋转, gt[4] 是 y 分辨率
        center_lon, center_lat = ds.xy(height // 2, width // 2)

    # 2. 解析 JSON 标签
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w, img_h = data.get('imageWidth', 1024), data.get('imageHeight', 1024)
    dates = re.findall(r'(\d{8})', data.get('imagePath', ''))
    
    result = {
        "metadata": {
            "imaging_time": dates[-1] if dates else "Unknown",
            "resolution": f"{data['shapes'][0].get('RES', '')}m",
            "center_coordinates": {"longitude": round(center_lon, 6), "latitude": round(center_lat, 6)}
        },
        "scene_context": {
            "scene_type": "",
            "time_of_day": "",
            "weather_conditions": "",
            "background_elements": [],
            "arrangement": "",
            "detail_description": ""
        },
        "objects_enrichment": {}
    }

    # 3. 转换坐标
    for i, shape in enumerate(data['shapes']):
        pts = shape['points']
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        x_c, y_c = (min_x + max_x) / 2, (min_y + max_y) / 2
        
        # 计算绝对经纬度
        lon, lat = ds.xy(y_c, x_c)
        
        ship_id = f"Ship_{i+1:03d}"
        result["objects_enrichment"][ship_id] = {
            "class": shape.get('label', 'Unknown'),
            "position": [round(x_c/img_w, 6), round(y_c/img_h, 6), round((max_x-min_x)/img_w, 6), round((max_y-min_y)/img_h, 6)],
            "abs_coordinates": {"longitude": round(lon, 6), "latitude": round(lat, 6)},
            "visual_appearance": "", "activity_status": "", "immediate_surroundings": ""
        }
    return result

def format_ship_spatial_text(analysis_results, top_k=8):
    """
    根据 position [x, y] 计算空间量化结果
    每艘船只仅保留距离最近的 top_k 艘船
    明确方向语义：other ship 相对于当前 ship（无歧义）
    返回 {ship_id: 单船量化文本} 字典
    """
    ship_spatial_dict = {}
    items = list(analysis_results.items())

    for i, (ship_id, info) in enumerate(items):
        x, y = info['position'][0], info['position'][1]

        # 计算绝对位置
        h_pos = "Left" if x < 0.33 else "Right" if x > 0.66 else "Center"
        v_pos = "Top" if y < 0.33 else "Bottom" if y > 0.66 else "Middle"

        line = f"--- {ship_id} ---\n"
        line += f"Global Position: {h_pos}-{v_pos}\n"
        line += f"Relative Relations (Top-{top_k} Neighbors):\n"

        # 计算与其他所有舰船的距离
        relations = []
        for j, (other_id, other_info) in enumerate(items):
            if i == j:
                continue

            ox, oy = other_info['position'][0], other_info['position'][1]
            dx, dy = ox - x, oy - y
            dist = (dx ** 2 + dy ** 2) ** 0.5

            relations.append((dist, other_id, dx, dy))

        # 按距离排序并取前 top_k
        relations.sort(key=lambda x: x[0])

        for dist, other_id, dx, dy in relations[:top_k]:
            d_label = (
                "Very Close" if dist < 0.1 else
                "Close" if dist < 0.3 else
                "Moderate" if dist < 0.6 else
                "Far"
            )

            vertical = "Down" if dy > 0 else "Up"
            horizontal = "Right" if dx > 0 else "Left"
            dir_label = f"{vertical}-{horizontal}"

            # ⭐ 核心修改：显式写 relative to
            line += (
                f"  - Relative to {ship_id}, "
                f"{other_id} is {d_label} (dist: {dist:.2f}) "
                f"at the {dir_label}.\n"
            )

        ship_spatial_dict[ship_id] = line

    return ship_spatial_dict
