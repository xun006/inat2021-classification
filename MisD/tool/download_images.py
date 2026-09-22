import os
import random
import shutil
import sys

# 配置参数（请根据需要修改）
SOURCE_BASE = "/mnt/hdd8t/Mingle/xyyy/data/inat2021/plants/val"
CATEGORY = "05961_Plantae_Tracheophyta_Liliopsida_Asparagales_Iridaceae_Crocus_tommasinianus"
DEST_DIR = "/mnt/hdd8t/Mingle/xyyy/selected_images"          # 保存图片的目标文件夹
NUM_IMAGES = 5                          # 要抽取的图片数量

# 支持的图片扩展名（可根据实际情况增减）
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def main():
    # 构造类别路径
    category_path = os.path.join(SOURCE_BASE, CATEGORY)
    if not os.path.isdir(category_path):
        print(f"错误：类别文件夹不存在 -> {category_path}")
        sys.exit(1)

    # 收集该类别下所有图片文件
    all_files = []
    for fname in os.listdir(category_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            all_files.append(fname)

    if not all_files:
        print(f"警告：类别文件夹中没有找到图片文件 -> {category_path}")
        return

    # 随机抽取（若不足则全部取出）
    sample_count = min(NUM_IMAGES, len(all_files))
    selected = random.sample(all_files, sample_count)

    # 创建目标文件夹
    os.makedirs(DEST_DIR, exist_ok=True)

    # 复制图片
    for fname in selected:
        src = os.path.join(category_path, fname)
        dst = os.path.join(DEST_DIR, fname)
        shutil.copy2(src, dst)
        print(f"已复制：{fname}")

    print(f"\n完成！共复制 {len(selected)} 张图片到 {DEST_DIR}")

if __name__ == "__main__":
    main()