import numpy as np

def count_rmsd(str_img, orig_img):
    # 确保两个输入是numpy数组
    str_img = np.array(str_img)
    orig_img = np.array(orig_img)
    
    # 计算两个图像之间的差异
    diff = str_img - orig_img
    
    # 计算差异的平方
    squared_diff = np.square(diff)
    
    # 计算平方差的平均值
    mean_squared_diff = np.mean(squared_diff)
    
    # 计算RMSD
    rmsd = np.sqrt(mean_squared_diff)
    
    return rmsd,diff
def count_masked_rmsd(str_img, orig_img, mask_img):
    # 确保三个输入都是numpy数组
    str_img = np.array(str_img)
    orig_img = np.array(orig_img)
    mask_img = np.array(mask_img)

    # 检查维度是否相等
    if str_img.shape != orig_img.shape or str_img.shape != mask_img.shape:
        raise ValueError("输入图像的维度必须相等")

    # 计算差异
    diff = str_img - orig_img

    # 创建一个与diff相同形状的数组，初始值为0
    masked_diff = np.zeros_like(diff)

    # 填充掩膜位置的差异
    masked_diff[mask_img >0.12] = diff[mask_img >0.12]

    # 计算平方差
    squared_diff = np.square(masked_diff)

    # 计算平方差的平均值，只在掩膜位置
    mean_squared_diff = np.mean(squared_diff[mask_img >0.12])

    # 计算RMSD
    rmsd = np.sqrt(mean_squared_diff)

    return rmsd, masked_diff
