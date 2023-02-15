import numpy as np
import random


def prior_mix(source_image, source_label, target_image, target_label,  prob=0.5, max_mix_num=1, cut_rate=0.25, ):

    # print(random.random())
    # mask =
    source_center_index = np.argwhere(source_label != 0)
    target_center_index = np.argwhere(target_label != 0)
    # size = source_label.shape
    bbox_coords = []
    mix_patch = []
    
    for n in range(max_mix_num):
        # 正方形区域中心点随机出现
        W, H = source_label.shape
        cut_rate = np.sqrt(cut_rate)
        cut_w = np.uint16(W * cut_rate)
        cut_h = np.uint16(H * cut_rate)
        if random.random() < prob:

            s_bbx1, s_bby1, s_bbx2, s_bby2 = rand_bbox_2d(
                (W, H), (cut_w, cut_h), source_center_index)  # 随机产生一个box的四个坐标
            t_bbx1, t_bby1, t_bbx2, t_bby2 = rand_bbox_2d(
                (W, H), (cut_w, cut_h), target_center_index)

            if s_bbx2 - s_bbx1 > t_bbx2 - t_bbx1:
                s_bbx1, s_bbx2 = adjust_bbox(s_bbx1, s_bbx2, t_bbx1, t_bbx2)
            elif s_bbx2 - s_bbx1 < t_bbx2 - t_bbx1:
                t_bbx1, t_bbx2 = adjust_bbox(t_bbx1, t_bbx2, s_bbx1, s_bbx2)

            if s_bby2 - s_bby1 > t_bby2 - t_bby1:
                s_bby1, s_bby2 = adjust_bbox(s_bby1, s_bby2, t_bby1, t_bby2)
            elif s_bby2 - s_bby1 < t_bby2 - t_bby1:
                t_bby1, t_bby2 = adjust_bbox(t_bby1, t_bby2, s_bby1, s_bby2)

            source_image[s_bbx1:s_bbx2,
                         s_bby1:s_bby2] = target_image[t_bbx1:t_bbx2, t_bby1:t_bby2]
            source_label[s_bbx1:s_bbx2,
                         s_bby1:s_bby2] = target_label[t_bbx1:t_bbx2, t_bby1:t_bby2]
            mix_label = target_label[t_bbx1:t_bbx2, t_bby1:t_bby2]
        else:

            s_bbx1, s_bby1, s_bbx2, s_bby2 = 0, 0, 0, 0
            mix_label = np.zeros((cut_w//2*2, cut_h//2*2))

        bbox_coords.append([[s_bbx1, s_bby1], [s_bbx2, s_bby2]])
        mix_patch.append(mix_label)

    return source_image, source_label, bbox_coords, mix_patch


def label_mix_with_patch(label, cut_patch, bbox_coords):

    for i in range(cut_patch.shape[0]):
        for j in range(cut_patch.shape[1]):

            if not (bbox_coords[i][j][0][0] == 0 and bbox_coords[i][j][1][0] == 0):
                s_bbx1, s_bbx2 = int(bbox_coords[i][j][0][0].tolist()), int(
                    bbox_coords[i][j][1][0].tolist())
                s_bby1, s_bby2 = int(bbox_coords[i][j][0][1].tolist()), int(
                    bbox_coords[i][j][1][1].tolist())
                label[i, s_bbx1:s_bbx2, s_bby1:s_bby2] = cut_patch[i, j, :, :]

    return label


def rand_bbox_2d(image_size, cut_size, center_index=None):
    W, H = image_size
    cut_w, cut_h = cut_size

    # uniform
    if center_index is None or len(center_index) == 0:
        cx = np.random.randint(cut_w//2,W-cut_w//2)
        cy = np.random.randint(cut_h//2,H-cut_h//2)
    else:
  
        random_index = np.random.randint(len(center_index))
        cx, cy = center_index[random_index]
        # cy = center_index[random_index][1]
    if cx - cut_w // 2 < 0:
        cx = cut_w // 2
    elif cx + cut_w // 2 > W:
        cx = W - cut_w // 2

    if cy - cut_h // 2 < 0:
        cy = cut_h // 2
    elif cy + cut_h // 2 > H:
        cy = H - cut_h // 2

    bbx1 = cx - cut_w // 2
    bby1 = cy - cut_h // 2
    bbx2 = cx + cut_w // 2
    bby2 = cy + cut_h // 2

    return bbx1, bby1, bbx2, bby2


def adjust_bbox(large_lower, large_uper, small_lower, small_uper):
    half_size = (small_uper - small_lower) // 2
    center = large_lower + (large_uper - large_lower) // 2
    new_large_lower = center-half_size
    new_large_uper = center+half_size

    return new_large_lower, new_large_uper



