import utils.Draw as Draw
import numpy as np
import utils.Config as Config
def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength,arg_random_nails = None,mask_pic = None):

    best_cumulative_improvement = -999999
    best_nail_position = None
    best_nail_idx = None

     # 计算钉子到当前点的距离
    distances = np.linalg.norm(nails - current_position, axis=1)


    if Config.USE_RANDOM_NAILS and Config.RANDOM_NAILS_NUM != None :
        closest_nail_indices = np.argsort(distances)[:arg_random_nails]
        nails_and_ids = list(zip(closest_nail_indices, nails[closest_nail_indices]))
    else:
        nails_and_ids = enumerate(nails)


    for nail_idx, nail_position in nails_and_ids:

        overlayed_line, rr, cc = Draw.get_aa_line(current_position, nail_position, str_strength, str_pic)
        if(Config.USE_MASK):
            before_overlayed_line_diff = (np.abs(str_pic[rr, cc] - orig_pic[rr, cc])*(mask_pic[rr, cc]))**2
            after_overlayed_line_diff = (np.abs(overlayed_line - orig_pic[rr, cc])*(mask_pic[rr, cc]))**2
        else:
            before_overlayed_line_diff = np.abs(str_pic[rr, cc] - orig_pic[rr, cc])**2
            after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rr, cc])**2
        cumulative_improvement =  np.sum(before_overlayed_line_diff - after_overlayed_line_diff)

        if cumulative_improvement >= best_cumulative_improvement:
            best_cumulative_improvement = cumulative_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cumulative_improvement