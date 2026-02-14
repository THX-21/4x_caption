import json
import os
from PIL import Image, ImageDraw, ImageFont

def visualize_ships(seq):
    json_path = f"data/metadata/train/result_{seq}.json"
    img_path = f"/root/autodl-fs/data/imgs/train/rgb_images/{seq}.jpg"
    output_path = f"data/viz/train/labeled_{seq}.png" # 导出为 PNG 方便查看

    if not os.path.exists(json_path) or not os.path.exists(img_path):
        print(f"错误: 找不到 {json_path} 或图像文件")
        return

    # 1. 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 打开图像并创建画笔
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # 3. 尝试加载字体，如果失败则使用默认字体
    try:
        # macOS 常用字体路径，Windows 可改为 "arial.ttf"
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()

    # 4. 遍历并绘制
    print(f"正在处理图像 {img_path} ({w}x{h})...")
    for ship_id, info in data['objects_enrichment'].items():
        xc, yc, sw, sh = info['position']
        
        # 归一化坐标转像素坐标
        left = (xc - sw/2) * w
        top = (yc - sh/2) * h
        right = (xc + sw/2) * w
        bottom = (yc + sh/2) * h
        
        # 绘制矩形框 (红色，宽度 3)
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        
        # 绘制标签背景
        text_bbox = draw.textbbox((left, top), ship_id, font=font)
        draw.rectangle([text_bbox[0], text_bbox[1]-2, text_bbox[2]+4, text_bbox[3]+2], fill="red")
        
        # 绘制文字 (白色)
        draw.text((left + 2, top - 2), ship_id, fill="white", font=font)
        print(f"标注完成: {ship_id}")

    # 5. 保存结果
    img.save(output_path)
    print(f"\n结果已保存至: {output_path}")

if __name__ == "__main__":
    [visualize_ships(f"{i:05d}") for i in range(1, 91)]