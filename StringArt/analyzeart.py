
import utils.ImageProcessing as ImageProcessing
import utils.CustomFile as CustomFile
import utils.PosGenerate as PosGenerate
import utils.Draw as Draw
import matplotlib.image as mpimg
import utils.CountLoss as CountLoss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == "__main__":
    
    coordinates_norm,order = CustomFile.read_seq_file("output\\1_mask.seq")
    orig_pic = ImageProcessing.largest_square(ImageProcessing.img_normalize(ImageProcessing.rgb2gray(np.array(mpimg.imread("input\\useimg\\1.jpg")))))
    
    output_image_dimens = orig_pic.shape
    blank = Draw.init_canvas(output_image_dimens, black=False)
    coordinates = PosGenerate.expand_normlized_coordinates(coordinates_norm,orig_pic.shape[0])
    str_pic = Draw.pull_order_to_array_bw(
            order,
            blank,
            coordinates,
            -0.1
        )
    rmsd,diff = CountLoss.count_rmsd(str_pic, orig_pic)
    print(rmsd)

      # 显示原始图片和生成图片
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(orig_pic, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Generated Image")
    plt.imshow(str_pic, cmap='gray')
    plt.axis('off')

    # 绘制热力图以展示diff
    plt.subplot(1, 3, 3)
    plt.title("Difference Heatmap")
    #sns.heatmap(diff, cmap='coolwarm', center=0, vmin=-1, vmax=1, cbar=False)
    plt.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis('off')

    plt.show()
    
