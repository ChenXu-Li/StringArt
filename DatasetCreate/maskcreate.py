import os
import cv2
import numpy as np

def edge_detection(input_folder, output_folder):
    # 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构建完整文件路径
            file_path = os.path.join(input_folder, filename)
            print(f"处理文件: {file_path}")

            # 读取图片
            image = cv2.imread(file_path)
            if image is None:
                print(f"无法读取文件: {file_path}")
                continue

            # 转换为灰度图
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 进行边缘检测
            edges = cv2.Canny(gray_image, 100, 200)

            # 保存为掩模图片
            output_path = os.path.join(output_folder, f"mask_{filename}")
            cv2.imwrite(output_path, edges)
            print(f"保存掩模文件: {output_path}")

if __name__ == "__main__":
    # 指定输入和输出文件夹路径
    input_folder = "D:\littlecode\ART\DatasetCreate\input\cutimgs"
    output_folder = "D:\littlecode\ART\DatasetCreate\input\mask"
    
    # 调用边缘检测函数
    edge_detection(input_folder, output_folder)