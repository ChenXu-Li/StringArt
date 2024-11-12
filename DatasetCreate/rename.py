import os

def rename_files_in_folder(folder_path, file_extension, new_name_format):
    # 遍历指定文件夹中的所有文件
    for count, filename in enumerate(os.listdir(folder_path)):
        if filename.lower().endswith(file_extension):
            # 构建完整的文件路径
            old_file_path = os.path.join(folder_path, filename)

            # 生成新的文件名
            new_filename = new_name_format.format(count) + file_extension
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"重命名: {old_file_path} -> {new_file_path}")

if __name__ == "__main__":
    # 输入文件夹路径和文件后缀
    folder_path = "D:\littlecode\ART\DatasetCreate\input\cutimgs"
    file_extension = ".png"
    new_name_format = "cutgrey_{:04d}"

    # 调用重命名函数
    rename_files_in_folder(folder_path, file_extension, new_name_format)