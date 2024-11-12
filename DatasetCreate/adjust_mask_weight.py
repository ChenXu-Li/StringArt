import os
from PIL import Image
import numpy as np

def trans_weight(w):
    epsilon = 1e-6  # 容忍值
    if abs(w - 0.4) < epsilon:
        return 0.8
    elif abs(w - 0.2) < epsilon:
        return 0.2
    elif abs(w - 0.6) < epsilon:
        return 0.9
    elif abs(w - 1.0) < epsilon:
        return 0.85
    elif abs(w - 0) < epsilon:
        return 0.1
    else:
        return w  # 返回原值，处理未覆盖的情况

def adjust_masks(input_folder, output_folder):
    # 创建输出文件夹，如果不存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # 仅处理图片文件
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path).convert("L")  # 转换为灰度图

            # 将图像数据转换为numpy数组
            img_array = np.array(image) / 255.0  # 归一化到[0, 1]

            # 应用像素重新映射
            mapped_array = np.vectorize(trans_weight)(img_array)

            # 将映射后的数组转换回图像
            new_image = Image.fromarray((mapped_array * 255).astype(np.uint8))

            # 保存新的掩码图像
            new_image.save(os.path.join(output_folder, filename))
            print(f"Processed and saved: {filename}")

if __name__ == "__main__":
    input_folder = "D:\\littlecode\\ART\\StringArt\\input\\usedmasks"
    output_folder = "D:\\littlecode\\ART\\DatasetCreate\\input\\adjust_masks"
    adjust_masks(input_folder, output_folder)
    print("调整完成！")