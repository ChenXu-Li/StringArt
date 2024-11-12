import os
from PIL import Image

def process_images(input_folder, output_folder, size):
    # 创建输出文件夹，如果不存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构建完整的文件路径
            file_path = os.path.join(input_folder, filename)
            print(f"处理文件: {file_path}")

            # 打开图片
            with Image.open(file_path) as img:
                # # 裁剪为正方形
                # min_dimension = min(img.size)
                # left = (img.width - min_dimension) / 2
                # top = (img.height - min_dimension) / 2
                # right = (img.width + min_dimension) / 2
                # bottom = (img.height + min_dimension) / 2

                # img_cropped = img.crop((left, top, right, bottom))

                # 缩放到指定分辨率
                # img_resized = img_cropped.resize(size, Image.ANTIALIAS)
                # img_resized = img.resize(size, Image.ANTIALIAS)
                img_resized = img.resize(size, Image.NEAREST)

                # 转换为灰度图片
                # img_gray = img_resized.convert('L')

                # 调暗 90%
                # img_darkened = Image.eval(img_gray, lambda x: x * 0.9)

                 # 修改文件名，将前缀从五位数字改为纯数字
                # 提取前缀并去掉前导零
                # prefix = filename.split('_')[0].lstrip('0')
                # if not prefix:  # 如果前缀为空，设为 '0'
                #     prefix = '0'
                prefix = filename.split('.')[0]
            
                
                new_filename = f"{prefix}.jpg"  # 生成新的文件名

                if new_filename.startswith('.'):  # 处理没有数字的情况
                    new_filename = '0' + new_filename
                    print(f"???????: {filename}")

                output_path = os.path.join(output_folder, new_filename)
                # 保存处理后的图片
                # output_path = os.path.join(output_folder, filename)
                # img_darkened.save(output_path)
                img_resized.save(output_path)
                print(f"保存文件: {output_path}")

if __name__ == "__main__":
    # 输入和输出文件夹路径
    # input_folder = input("请输入图片文件夹路径: ")
    # output_folder = input("请输入输出文件夹路径: ")
    # size = tuple(map(int, input("请输入目标分辨率 (宽 高，以空格分隔): ").split()))
    input_folder = "D:\littlecode\ART\DatasetCreate\input\\usemask"
    output_folder = "D:\littlecode\ART\DatasetCreate\input\\usedmasks"
    size = (2048, 2048)

    # 调用处理函数
    process_images(input_folder, output_folder, size)