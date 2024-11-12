import os
import numpy as np
import utils.PosGenerate as PosGenerate
#pos的区间是【0，1】
def write_pos_file(file_path, coordinates):
    # 检查文件后缀
    if not file_path.endswith('.pos'):
        raise ValueError("文件必须以 '.pos' 结尾")
    
    # 检查坐标值是否在 [0, 1] 范围内
    for x, y in coordinates:
        if not (0 <= x <= 1) or not (0 <= y <= 1):
            raise ValueError(f"坐标 ({x}, {y}) 超出范围 [0, 1]")
    
    # 获取绝对路径
    absolute_path = os.path.abspath(file_path)
    print(f"将坐标数据写入文件 {absolute_path}")
    
    # 打开文件并写入数据
    with open(file_path, 'w') as file:
        for x, y in coordinates:
            file.write(f'{x},{y}\n')

def read_pos_file(file_path):
    # 检查文件后缀
    if not file_path.endswith('.pos'):
        raise ValueError("文件必须以 '.pos' 结尾")
    
    # 获取绝对路径
    absolute_path = os.path.abspath(file_path)
    print(f"读取坐标数据 {absolute_path}")
    
    # 读取文件并将其转换为 numpy 数组
    with open(file_path, 'r') as file:
        coordinates = [list(map(float, line.strip().split(','))) for line in file]
    
    # 将读取的数据转换为 numpy 数组
    coordinates_array = np.array(coordinates)
    
    # 检查坐标值是否在 [0, 1] 范围内
    if not np.all((0 <= coordinates_array) & (coordinates_array <= 1)):
        raise ValueError("文件中的坐标值超出范围 [0, 1]")

    return coordinates_array

def write_seq_file(file_path, coordinates, order):
     # 检查文件后缀
    if not file_path.endswith('.seq'):
        raise ValueError("文件必须以 '.seq' 结尾")
    
    # 检查坐标值是否在 [0, 1] 范围内
    for x, y in coordinates:
        if not (0 <= x <= 1) or not (0 <= y <= 1):
            raise ValueError(f"坐标 ({x}, {y}) 超出范围 [0, 1]")
    
    # 获取绝对路径
    absolute_path = os.path.abspath(file_path)
    print(f"将坐标数据和连接顺序写入文件 {absolute_path}")
    
    with open(file_path, 'w') as file:
        for x, y in coordinates:
            file.write(f'{x},{y}\n')
        # 写入顺序信息
        file.write(f'# Order: {"-".join(map(str, order))}\n')
def read_seq_file(file_path):
    # 检查文件后缀
    if not file_path.endswith('.seq'):
        raise ValueError("文件必须以 '.seq' 结尾")
    
    # 获取绝对路径
    absolute_path = os.path.abspath(file_path)
    print(f"读取坐标数据和顺序信息 {absolute_path}")
    
    coordinates = []
    order = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('# Order:'):
                order = list(map(int, line.split(': ')[1].split('-')))
            else:
                x, y = map(float, line.split(','))
                coordinates.append((x, y))
   
    # 将读取的数据转换为 numpy 数组
    coordinates_array = np.array(coordinates)
    # 检查坐标值是否在 [0, 1] 范围内
    if not np.all((0 <= coordinates_array) & (coordinates_array <= 1)):
        raise ValueError("文件中的坐标值超出范围 [0, 1]")
    return np.array(coordinates), np.array(order)

if __name__ == "__main__":
    poslist = PosGenerate.create_evenly_spaced_nail_positions_normalization(100)
    write_pos_file("evenly_spaced.pos",poslist)
