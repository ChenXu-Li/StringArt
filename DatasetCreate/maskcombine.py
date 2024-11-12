import os
from PIL import Image
import numpy as np

def get_weight(file_name):
    if 'hair' in file_name:
        return 0.4
    elif 'cloth' in file_name:
        return 0.2
    elif 'hat' in file_name:
        return 0.2
    elif 'neck' in file_name:
        return 0.4
    elif 'skin' in file_name:
        return 0.6
    else:
        return 1.0

def combine_images(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有文件
    files = os.listdir(input_folder)

    # 按前缀分组
    grouped_images = {}
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):  # 根据需要添加其他格式
            prefix = file.split('_')[0]  # 获取前缀
            if prefix not in grouped_images:
                grouped_images[prefix] = []
            grouped_images[prefix].append(file)

     # 合并并保存图片
    for prefix, images in grouped_images.items():
        combined_mask = None
        for image_file in images:
            image_path = os.path.join(input_folder, image_file)
            img = Image.open(image_path).convert('L')  # 转换为灰度图像

            # 获取当前图片的权重
            weight = get_weight(image_file)
            print(f"当前图片的权重: {weight}")

            # 将图像转换为数组
            img_array = np.array(img) / 255.0  # 归一化到 [0, 1]

            if combined_mask is None:
                combined_mask = img_array * weight
            else:
                combined_mask += img_array * weight

        # 将合并后的结果归一化到 [0, 1]
        combined_mask = np.clip(combined_mask, 0, 1)

        # 转换回二值图像
        final_mask = (combined_mask * 255).astype(np.uint8)   # 大于0的部分设为255

        # 保存合并后的图片
        output_path = os.path.join(output_folder, f"{prefix}_combined.png")
        Image.fromarray(final_mask).save(output_path)
        print(f"保存合并后的图片: {output_path}")

if __name__ == "__main__":
    # input_folder = input("请输入要处理的文件夹路径: ")
    # output_folder = input("请输入保存合并后图片的文件夹路径: ")
    input_folder = "D:\littlecode\ART\DatasetCreate\input\\rawmasks"
    output_folder = "D:\littlecode\ART\DatasetCreate\input\\combinerawmasks"
    combine_images(input_folder, output_folder)
    print("合并完成！")