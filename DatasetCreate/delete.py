import os

def delete_non_png_files(folder_path):
    # 遍历指定文件夹
    for filename in os.listdir(folder_path):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 检查是否是文件且不是以 .png 结尾
        if os.path.isfile(file_path) and  filename.endswith('.pts'):
            print(f"删除文件: {file_path}")
            os.remove(file_path)

if __name__ == "__main__":
    # 指定要清理的文件夹路径
    folder_path = input("请输入要清理的文件夹路径: ")
    folder_path = "D:\littlecode\ART\DatasetCreate\\300W\\01_Indoor"
    
    # 调用函数
    delete_non_png_files(folder_path)