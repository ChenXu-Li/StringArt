import numpy as np
import matplotlib.pyplot as plt
def img_normalize(img):
    if img.max()>1:
        return img/255
    else:
        return img
def rgb2gray(image):
    # 检查图像的通道数
    if image.ndim == 3 and image.shape[2] == 3:  # 三通道彩色图像
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # 转换为灰度图
    elif image.ndim == 2:  # 已经是灰度图像
        return image  # 不变
    else:
        raise ValueError("输入图像必须是彩色或灰度图像")
def largest_square(image: np.ndarray) -> np.ndarray:
    short_edge = np.argmin(image.shape[:2])  # 0 = vertical <= horizontal; 1 = otherwise
    short_edge_half = image.shape[short_edge] // 2
    long_edge_center = image.shape[1 - short_edge] // 2

    # Adjust for odd dimensions to ensure square output
    if image.shape[short_edge] % 2 != 0:
        short_edge_half += 1
    
    if short_edge == 0:
        start = max(0, long_edge_center - short_edge_half)
        end = min(image.shape[1], long_edge_center + short_edge_half)
        return image[:, start:end]
    else:
        start = max(0, long_edge_center - short_edge_half)
        end = min(image.shape[0], long_edge_center + short_edge_half)
        return image[start:end, :]
def adjust_contrast_and_brightness(image, contrast_factor= 0.8,darkness_factor=0.8) :



    # 计算图像的平均亮度
    mean = np.mean(image)
    print(mean)
    # 调整对比度
    contrast_adjusted = (image - mean) * contrast_factor + mean
    darkness_adjusted = contrast_adjusted * darkness_factor
    # 确保值在[0, 1]范围内
    darkness_adjusted = np.clip(darkness_adjusted, 0, 1)
   
    # plt.figure(figsize=(2, 1), dpi=1024)
    # plt.subplot(1, 2, 1)
    
    # plt.imshow(contrast_adjusted, cmap='gray')
    # plt.clim(0, 1)
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    
    # plt.imshow(image, cmap='gray')
    # plt.clim(0, 1)
    # plt.axis('off')
    # plt.show()
    return darkness_adjusted
