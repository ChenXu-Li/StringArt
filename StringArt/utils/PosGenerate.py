import os
import numpy as np
from skimage.draw import ellipse_perimeter
from math import atan2
def create_rectangle_nail_positions(shape, nail_step=2):
    height, width = shape

    nails_top = [(0, i) for i in range(0, width, nail_step)]
    nails_bot = [(height-1, i) for i in range(0, width, nail_step)]
    nails_right = [(i, width-1) for i in range(1, height-1, nail_step)]
    nails_left = [(i, 0) for i in range(1, height-1, nail_step)]
    nails = nails_top + nails_right + nails_bot + nails_left

    return np.array(nails)
def create_rectangle_nail_positions_normalization(single_num=100):
    step = 1/single_num

    nails_top = [(0, i*step) for i in range(0, single_num+1)]
    nails_bot = [(1, i*step) for i in range(0, single_num+1)]
    nails_right = [(i*step, 1) for i in range(1, single_num)]
    nails_left = [(i*step, 0) for i in range(1, single_num)]
    nails = nails_top + nails_right + nails_bot + nails_left

    return np.array(nails)

def create_circle_nail_positions(shape, nail_step=2, r1_multip=1, r2_multip=1):
    height = shape[0]
    width = shape[1]

    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 1
    rr, cc = ellipse_perimeter(centre[0], centre[1], int(radius*r1_multip), int(radius*r2_multip))
    nails = list(set([(rr[i], cc[i]) for i in range(len(cc))]))
    nails.sort(key=lambda c: atan2(c[0] - centre[0], c[1] - centre[1]))
    nails = nails[::nail_step]

    return np.asarray(nails)

def create_circle_nail_positions_normalization(single_num=100):
    centre = (0.5, 0.5)  # 中心点坐标
    radius = 0.5  # 半径
    
    # 计算每个点的角度间隔
    angle_step = 2 * np.pi / single_num
    
    # 生成圆周上的坐标点
    nails = []
    for i in range(single_num):
        angle = i * angle_step
        x = centre[0] + radius * np.cos(angle)
        y = centre[1] + radius * np.sin(angle)
        nails.append((y, x))  # 注意这里的顺序，因为通常图像坐标系是以 (y, x) 表示
    return np.asarray(nails)
def create_random_nail_positions_normalization(nail_num=100):
    # 生成随机坐标点
    nails = np.random.rand(nail_num, 2)
    
    # 归一化坐标点到 [0, 1] 范围内
    nails = nails.clip(0, 1)
    
    return nails

def create_gaussian_nail_positions_normalization(nail_num=100):
    # 生成标准正态分布的随机坐标点
    nails = np.random.randn(nail_num, 2)
    
    # 将坐标点归一化到 [0, 1] 范围内
    nails = (nails - nails.min(axis=0)) / (nails.max(axis=0) - nails.min(axis=0))
    
    # 确保所有坐标都在 [0, 1] 范围内
    nails = nails.clip(0, 1)
    
    return nails
def create_evenly_spaced_nail_positions_normalization(nail_num=100):
    # 计算每个轴上需要的点的数量
    points_per_axis = int(np.sqrt(nail_num))
    
    # 生成 [0, 1] 范围内等距分布的点
    x = np.linspace(0, 1, points_per_axis)
    y = np.linspace(0, 1, points_per_axis)
    
    # 创建网格点 (x, y)
    xv, yv = np.meshgrid(x, y)
    
    # 将网格点展开成坐标列表
    nails = np.column_stack([xv.ravel(), yv.ravel()])
    
    # 如果生成的点比 nail_num 多，取前 nail_num 个
    return nails[:nail_num]

def expand_normlized_coordinates(coordinates,w):#从（0，1）坐标转换为（0，w）坐标
    return (coordinates * (w-1)).astype(int)
def scale_nails(x_ratio, y_ratio, nails):
    return [(int(y_ratio*nail[0]), int(x_ratio*nail[1])) for nail in nails]

if __name__ == "__main__":

    pass