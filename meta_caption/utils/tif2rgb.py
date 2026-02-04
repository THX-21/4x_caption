import os, glob
from PIL import Image

def convert(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in glob.glob(os.path.join(src_dir, "*.tif")):
        name = os.path.splitext(os.path.basename(f))[0] + ".jpg"
        Image.open(f).convert("RGB").save(os.path.join(dst_dir, name))

if __name__ == "__main__":
    convert("/autodl-fs/data/RGB/test/images", "/autodl-fs/data/RGB/test/rgb_images")
