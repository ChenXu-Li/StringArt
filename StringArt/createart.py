from time import time
import os
import numpy as np
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import utils.ImageProcessing as ImageProcessing

import utils.PosGenerate as PosGenerate
import utils.Draw as Draw
import utils.CustomFile as CustomFile
import utils.GreddySearch as GreddySearch
import utils.Config as Config

import utils.CountLoss as CountLoss

global every_time
def create_art(nails, orig_pic, str_pic, str_strength, i_limit=None,mask_pic=None,random_num=None):
    global every_time
    start = time()
    iter_times = []
 
    current_position = nails[0]
    pull_order = [0]

    i = 0
    fails = 0
    while True:
        start_iter = time()

        i += 1
        
        if i%500 == 0:
            print(f"Iteration {i}")
            print(f"Fail_num {fails}/{i}")

        
        if i_limit == None:
            if fails >= 3:
                break
        else:
            if i > i_limit:
                break

        idx, best_nail_position, best_cumulative_improvement = GreddySearch.find_best_nail_position(current_position, nails,
                                                                                       str_pic, orig_pic, str_strength,random_num,mask_pic)
        

        if best_cumulative_improvement <= 0:
            fails += 1
            continue

        pull_order.append(idx)
        best_overlayed_line, rr, cc = Draw.get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line

        current_position = best_nail_position
        iter_times.append(time() - start_iter)

    print(f"Time: {time() - start}")
    every_time = time() - start
    print(f"Avg iteration time: {np.mean(iter_times)}")
    return pull_order
def one_art(index):

    

    side_len = Config.SIDE_LEN #输出图片结果的分辨率

    export_strength = Config.EXPORT_STRENGTH#线的颜色强度

    pull_amount = Config.PULL_AMOUNT#迭代次数上限

    random_nails = Config.RANDOM_NAILS_NUM #是否随机nail选择来加快贪婪算法

    nail_num = Config.NAIL_NUM

    #1.加载并且归一化图片数据
    input_file = os.path.join("input", "usedimages", f"{index}.jpg")
    input_mask_file = os.path.join("input", "adjustmasks", f"{index}.jpg")
    img = mpimg.imread(input_file)
    img = ImageProcessing.rgb2gray(img)
    img = ImageProcessing.img_normalize(img)#归一化到01
    img = ImageProcessing.adjust_contrast_and_brightness(img)
    mask_img = None
    mask_img = mpimg.imread(input_mask_file)
    mask_img = ImageProcessing.img_normalize(mask_img)
    shape = ( len(img), len(img[0]) )
    print("input_image_shape:", shape)
    output_image_dimens = int(side_len), int(side_len)
    print("output_image_dimens:", output_image_dimens)
        
    #2.生成钉子坐标并适应输入图片分辨率
    nails_norm = PosGenerate.create_evenly_spaced_nail_positions_normalization(nail_num)
    nails = PosGenerate.expand_normlized_coordinates(nails_norm,shape[0])

    #3.贪婪算法找出连接顺序
    # orig_pic = ImageProcessing.rgb2gray(img)
    orig_pic = img


    str_pic = Draw.init_canvas(shape, black=False)
    pull_order = create_art(nails, orig_pic, str_pic, -0.05, i_limit=pull_amount,mask_pic=mask_img,random_num=random_nails)

    #4.根据钉子坐标和连接顺序生成输出图片(图片还是归一化状态)
    blank = Draw.init_canvas(output_image_dimens, black=False)
    scaled_nails = PosGenerate.scale_nails(
        output_image_dimens[1] / shape[1],
        output_image_dimens[0] / shape[0],
        nails
    )
    results = []
    for i in [int(Config.PULL_AMOUNT*0.2),int(Config.PULL_AMOUNT*0.5),Config.PULL_AMOUNT]:
        result = Draw.pull_order_to_array_bw(
            pull_order[:i],
            blank,
            scaled_nails,
            -export_strength
        )
        rmsd, masked_diff = CountLoss.count_masked_rmsd(result, orig_pic, mask_img)
        results.append((result,rmsd,masked_diff))
    return results,orig_pic,mask_img

    #保存seq结果
    # output_name=f"{str(index)}_P{Config.PULL_AMOUNT}_N{Config.NAIL_NUM}" if(Config.USE_MASK) else f"{str(index)}_P{Config.PULL_AMOUNT}_N{Config.NAIL_NUM}_MASK"
    # CustomFile.write_seq_file("output\\"+output_name+".seq", nails_norm, pull_order)
    return result

def save_fig(results_1, o, m, results_2, output_file):
    global every_time
    plt.figure(figsize=(7, 2), dpi=1024)  # 设置图形尺寸

    # 第一行子图
    plt.subplot(2, 7, 1)
    plt.imshow(o, cmap='gray',vmin=0, vmax=1)
    plt.axis('off')
    plt.text(0.5, -0.1, "a) Original Image", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 2)
    plt.imshow(results_1[0][0], cmap='gray')
    plt.axis('off')
    plt.text(0.5, -0.1, f"b){int(Config.PULL_AMOUNT*0.2)}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 3)
    plt.imshow(results_1[0][2], cmap='coolwarm', vmin=-1, vmax=1)
   # cbar = plt.colorbar(shrink=0.5)  # 调整 colorbar 的大小
    #cbar.ax.tick_params(labelsize=3)  # 设置 colorbar 字体大小
    plt.axis('off')
    plt.text(0.5, -0.1, f"c) Masked RMSD: {results_1[0][1]:.4f}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 4)
    plt.imshow(results_1[1][0], cmap='gray')
    plt.axis('off')
    plt.text(0.5, -0.1, f"d) {int(Config.PULL_AMOUNT*0.5)}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 5)
    plt.imshow(results_1[1][2], cmap='coolwarm', vmin=-1, vmax=1)
   # cbar = plt.colorbar(shrink=0.5)
    #cbar.ax.tick_params(labelsize=3)
    plt.axis('off')
    plt.text(0.5, -0.1, f"e) Masked RMSD: {results_1[1][1]:.4f}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 6)
    plt.imshow(results_1[2][0], cmap='gray')
    plt.axis('off')
    plt.text(0.5, -0.1, f"f) {Config.PULL_AMOUNT}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 7)
    plt.imshow(results_1[2][2], cmap='coolwarm', vmin=-1, vmax=1)
   # cbar = plt.colorbar(shrink=0.5)
    #cbar.ax.tick_params(labelsize=3)
    plt.axis('off')
    plt.text(0.5, -0.1, f"g) Masked RMSD: {results_1[2][1]:.4f}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    # 第二行子图
    plt.subplot(2, 7, 8)
    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.text(0.5, -0.1, "h) Mask Image", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 9)
    plt.imshow(results_2[0][0], cmap='gray')
    plt.axis('off')
    plt.text(0.5, -0.1, f"i) {int(Config.PULL_AMOUNT*0.2)}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 10)
    plt.imshow(results_2[0][2], cmap='coolwarm', vmin=-1, vmax=1)
   # cbar = plt.colorbar(shrink=0.5)
    #cbar.ax.tick_params(labelsize=3)
    plt.axis('off')
    plt.text(0.5, -0.1, f"j) Masked RMSD: {results_2[0][1]:.4f}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 11)
    plt.imshow(results_2[1][0], cmap='gray')
    plt.axis('off')
    plt.text(0.5, -0.1, f"k) {int(Config.PULL_AMOUNT*0.5)} ", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 12)
    plt.imshow(results_2[1][2], cmap='coolwarm', vmin=-1, vmax=1)
   # cbar = plt.colorbar(shrink=0.5)
    #cbar.ax.tick_params(labelsize=3)
    plt.axis('off')
    plt.text(0.5, -0.1, f"l) Masked RMSD: {results_2[1][1]:.4f}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 13)
    plt.imshow(results_2[2][0], cmap='gray')
    plt.axis('off')
    plt.text(0.5, -0.1, f"m) {Config.PULL_AMOUNT}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)

    plt.subplot(2, 7, 14)
    plt.imshow(results_2[2][2], cmap='coolwarm', vmin=-1, vmax=1)
   # cbar = plt.colorbar(shrink=0.5)
    #cbar.ax.tick_params(labelsize=3)
    plt.axis('off')
    plt.text(0.5, -0.1, f"n) Masked RMSD: {results_2[2][1]:.4f}", ha='center', va='center', fontsize=3, transform=plt.gca().transAxes)
    # 添加配置参数信息
    plt.figtext(0.5, 0.05, 
            f"PULL_AMOUNT = {Config.PULL_AMOUNT}    "
            f"NAIL_NUM = {Config.NAIL_NUM}    "
            f"USE_MASK = {Config.USE_MASK}    "
            f"USE_RANDOM_NAILS = {Config.USE_RANDOM_NAILS}    "
            f"RANDOM_NAILS_NUM = {Config.RANDOM_NAILS_NUM}    "
            f"EXPORT_STRENGTH = {Config.EXPORT_STRENGTH}    "
            f"SIDE_LEN = {Config.SIDE_LEN}    " 
            f"Time = {every_time}", 
            ha='center', va='center', fontsize=4)
    # 调整子图间距
    # 

    plt.tight_layout(w_pad=0.1,h_pad=1.08,rect=[0, 0.1, 1, 1]) # 自动调整子图间距
    plt.subplots_adjust(wspace=0.05)  # 调整子图之间的水平和垂直间距
    plt.savefig(output_file, dpi=256)  # 保存为文件
    plt.close()  # 关闭图形
if __name__ == '__main__':
    random_nail_n=[100,200,400,500]
    line_strength=[0.2,0.5,0.7,1.0]
    for index in range(40,250):
        if(index%10==0):
            Config.RANDOM_NAILS_NUM = random_nail_n[int(index/10)%4]
        if(index%40==0):
            Config.EXPORT_STRENGTH = line_strength[int(index/40)%4]
        # output_file = "D:\littlecode\ART\StringArt\output\\"+str(index)+"_out.jpg"
        Config.USE_MASK=False
        results_1,o,m = one_art(index)
        print(f"RMSD{int(Config.PULL_AMOUNT*0.2)}:",results_1[0][1],f"RMSD{int(Config.PULL_AMOUNT*0.5)}:",results_1[1][1], f"RMSD{Config.PULL_AMOUNT}:",results_1[2][1])
        merged_result1 = np.hstack((o,results_1[0][0], results_1[0][2], results_1[1][0], results_1[1][2],results_1[2][0], results_1[2][2]))

        Config.USE_MASK=True
        results_2,o,m = one_art(index)
        merged_result2 = np.hstack((m,results_2[0][0], results_2[0][2], results_2[1][0], results_2[1][2],results_2[2][0], results_2[2][2]))


        merged_result = np.vstack((merged_result1, merged_result2))
        output_name=f"{str(index)}_out.jpg"

        output_file= os.path.join("output", output_name)
        # mpimg.imsave(output_file, merged_result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)
        # mpimg.imsave(output_file, merged_result, cmap=plt.get_cmap("coolwarm"), vmin=-1.0, vmax=1.0)
        save_fig(results_1,o,m,results_2,output_file)
        print("Saving result :"+output_file)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Create String Art')  
#     parser.add_argument('-i', action="store", dest="input_file",default="D:\littlecode\ART\StringArt\input\\usedimages\\1.jpg")
#     parser.add_argument('-im', action="store", dest="input_mask_file",default="D:\littlecode\ART\StringArt\input\\usedmasks\\1.png")
#     parser.add_argument('-o', action="store", dest="output_file", default="D:\littlecode\ART\StringArt\output\\1_out.jpg")
#     parser.add_argument('-d', action="store", type=int, dest="side_len", default=300)#输出图片结果的分辨率
#     parser.add_argument('-s', action="store", type=float, dest="export_strength", default=0.1)#线的颜色强度
#     parser.add_argument('-l', action="store", type=int, dest="pull_amount", default=500)#迭代次数上限
#     parser.add_argument('-r', action="store", type=int, dest="random_nails", default=None)#是否随机nail选择来加快贪婪算法
#     parser.add_argument('-r1', action="store", type=float, dest="radius1_multiplier", default=1)
#     parser.add_argument('-r2', action="store", type=float, dest="radius2_multiplier", default=1)
#     parser.add_argument('-n', action="store", type=int, dest="nail_num", default=600)#钉子数量
#     parser.add_argument('--wb', action="store_true")
#     parser.add_argument('--rgb', action="store_true")

#     args = parser.parse_args()


#     img = mpimg.imread(args.input_file)

#     img = ImageProcessing.img_normalize(img)#归一化到01

#     img = ImageProcessing.largest_square(img)

#     mask_img = None
#     if(Config.USE_MASK):
#         mask_img = mpimg.imread(args.input_mask_file)

#         mask_img = ImageProcessing.img_normalize(mask_img)

#         mask_img = ImageProcessing.largest_square(mask_img)

#     shape = ( len(img), len(img[0]) )
#     print("shape:", shape)
#     output_image_dimens = int(args.side_len), int(args.side_len)
#     print("output_image_dimens:", output_image_dimens)
        
#     nails_norm = PosGenerate.create_evenly_spaced_nail_positions_normalization(args.nail_num)
    
#     nails = PosGenerate.expand_normlized_coordinates(nails_norm,shape[0])

#     if args.rgb:
#         pass
#     else:
#         orig_pic = ImageProcessing.rgb2gray(img)*0.9

#         if args.wb:
#             str_pic = Draw.init_canvas(shape, black=True)
           
#             pull_order = create_art(nails, orig_pic, str_pic, 0.05, i_limit=args.pull_amount,mask_pic=mask_img)
#             blank = Draw.init_canvas(output_image_dimens, black=True)
#         else:
#             str_pic = Draw.init_canvas(shape, black=False)
#             pull_order = create_art(nails, orig_pic, str_pic, -0.05, i_limit=args.pull_amount,mask_pic=mask_img)
#             blank = Draw.init_canvas(output_image_dimens, black=False)

#         scaled_nails = PosGenerate.scale_nails(
#             output_image_dimens[1] / shape[1],
#             output_image_dimens[0] / shape[0],
#             nails
#         )

#         result = Draw.pull_order_to_array_bw(
#             pull_order,
#             blank,
#             scaled_nails,
#             args.export_strength if args.wb else -args.export_strength
#         )
#         CustomFile.write_seq_file("output\\1.seq", nails_norm, pull_order)

#         mpimg.imsave(args.output_file, result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)

#         print(f"Thread pull order by nail index:\n{'-'.join([str(idx) for idx in pull_order])}")


# #python createart.py -i D:\littlecode\PythonProjects\StringArt\test_data\a.jpg -o D:\littlecode\PythonProjects\StringArt\test_data\output.jpg




