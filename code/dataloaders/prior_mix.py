import numpy as np
import random
from scipy.ndimage import rotate, zoom
import torch

class PriorMixAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size, cur_rate=0.1, mix_prob=0.2, max_mix_num = 1):
        self.output_size = output_size
        self.cur_rate = cur_rate
        self.mix_prob = mix_prob
        self.max_mix_num = max_mix_num

    def __call__(self, image, label, mix_image, mix_label):

        image = self.resize(image)
        label = self.resize(label)
        mix_image = self.resize(mix_image)
        mix_label = self.resize(mix_label)
        # weak augmentation is rotation / flip
        image, label = random_rot_flip(image, label)
        mix_image, mix_label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        # image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # if idx in labeled_idxs:
        image, label, bbox_coords, mix_patchs = prior_mix(
            image, label, mix_image, mix_label, prob=self.mix_prob, max_mix_num=self.max_mix_num, cut_rate=self.cur_rate, )

        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        # fix dimensions
        label = torch.from_numpy(label.astype(np.uint8))
        bbox_coords = torch.from_numpy(
            np.array(bbox_coords).astype(np.float32))
        mix_patchs = torch.from_numpy(np.array(mix_patchs).astype(np.uint8))

        sample = {
            "image": image,
            "label": label,
            "bbox_coords": bbox_coords,
            "mix_patchs": mix_patchs
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

def prior_mix(source_image, source_label, target_image, target_label,  prob=0.5, max_mix_num=1, cut_rate=0.25, ):

    # print(random.random())
    # mask =
    source_center_index = np.argwhere(source_label != 0)
    target_center_index = np.argwhere(target_label != 0)
    # size = source_label.shape
    bbox_coords = []
    mix_patch = []

    # 正方形区域中心点随机出现\
    if len(source_label.shape) == 2:
        W, H = source_label.shape
        cut_rate = np.sqrt(cut_rate)
        cut_w = np.uint16(W * cut_rate)
        cut_h = np.uint16(H * cut_rate)
        for n in range(max_mix_num):
            if random.random() < prob:

                s_bbx1, s_bby1, s_bbx2, s_bby2 = rand_bbox(
                    (W, H), (cut_w, cut_h), source_center_index)  # 随机产生一个box的四个坐标
                t_bbx1, t_bby1, t_bbx2, t_bby2 = rand_bbox(
                    (W, H), (cut_w, cut_h), target_center_index)

                if s_bbx2 - s_bbx1 > t_bbx2 - t_bbx1:
                    s_bbx1, s_bbx2 = adjust_bbox(
                        s_bbx1, s_bbx2, t_bbx1, t_bbx2)
                elif s_bbx2 - s_bbx1 < t_bbx2 - t_bbx1:
                    t_bbx1, t_bbx2 = adjust_bbox(
                        t_bbx1, t_bbx2, s_bbx1, s_bbx2)

                if s_bby2 - s_bby1 > t_bby2 - t_bby1:
                    s_bby1, s_bby2 = adjust_bbox(
                        s_bby1, s_bby2, t_bby1, t_bby2)
                elif s_bby2 - s_bby1 < t_bby2 - t_bby1:
                    t_bby1, t_bby2 = adjust_bbox(
                        t_bby1, t_bby2, s_bby1, s_bby2)

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

    elif len(source_label.shape) == 3:
        W, H, S = source_label.shape
        cut_rate = pow(cut_rate, 1/3)
        cut_w = np.uint16(W * cut_rate)
        cut_h = np.uint16(H * cut_rate)
        cut_s = np.uint16(S * cut_rate)
        for n in range(max_mix_num):
            if random.random() < prob:

                s_bbx1, s_bby1, s_bbz1, s_bbx2, s_bby2, s_bbz2 = rand_bbox(
                    (W, H, S), (cut_w, cut_h, cut_s), source_center_index)  # 随机产生一个box的四个坐标
                t_bbx1, t_bby1, t_bbz1, t_bbx2, t_bby2, t_bbz2 = rand_bbox(
                    (W, H, S), (cut_w, cut_h, cut_s), target_center_index)

                if s_bbx2 - s_bbx1 > t_bbx2 - t_bbx1:
                    s_bbx1, s_bbx2 = adjust_bbox(
                        s_bbx1, s_bbx2, t_bbx1, t_bbx2)
                elif s_bbx2 - s_bbx1 < t_bbx2 - t_bbx1:
                    t_bbx1, t_bbx2 = adjust_bbox(
                        t_bbx1, t_bbx2, s_bbx1, s_bbx2)

                if s_bby2 - s_bby1 > t_bby2 - t_bby1:
                    s_bby1, s_bby2 = adjust_bbox(
                        s_bby1, s_bby2, t_bby1, t_bby2)
                elif s_bby2 - s_bby1 < t_bby2 - t_bby1:
                    t_bby1, t_bby2 = adjust_bbox(
                        t_bby1, t_bby2, s_bby1, s_bby2)

                if s_bbz2 - s_bbz1 > t_bbz2 - t_bbz1:
                    s_bbz1, s_bbz2 = adjust_bbox(
                        s_bbz1, s_bbz2, t_bbz1, t_bbz2)
                elif s_bbz2 - s_bbz1 < t_bbz2 - t_bbz1:
                    t_bbz1, t_bbz2 = adjust_bbox(
                        t_bbz1, t_bbz2, s_bbz1, s_bbz2)

                source_image[s_bbx1:s_bbx2,
                             s_bby1:s_bby2, s_bbz1:s_bbz2] = target_image[t_bbx1:t_bbx2, t_bby1:t_bby2, t_bbz1:t_bbz2]
                source_label[s_bbx1:s_bbx2,
                             s_bby1:s_bby2, s_bbz1:s_bbz2] = target_label[t_bbx1:t_bbx2, t_bby1:t_bby2, t_bbz1:t_bbz2]
                mix_label = target_label[t_bbx1:t_bbx2,
                                         t_bby1:t_bby2, t_bbz1:t_bbz2]
            else:

                s_bbx1, s_bby1, s_bbz1, s_bbx2, s_bby2, s_bbz2 = 0, 0, 0, 0, 0, 0
                mix_label = np.zeros((cut_w//2*2, cut_h//2*2, cut_s//2*2))

            bbox_coords.append(
                [[s_bbx1, s_bby1, s_bbz1], [s_bbx2, s_bby2, s_bbz2]])
            mix_patch.append(mix_label)

    return source_image, source_label, bbox_coords, mix_patch


def label_mix_with_patch(label, cut_patch, bbox_coords):
    if len(label.shape) == 3:
        for i in range(cut_patch.shape[0]):
            for j in range(cut_patch.shape[1]):

                if not (bbox_coords[i][j][0][0] == 0 and bbox_coords[i][j][1][0] == 0):
                    s_bbx1, s_bbx2 = int(bbox_coords[i][j][0][0].tolist()), int(
                        bbox_coords[i][j][1][0].tolist())
                    s_bby1, s_bby2 = int(bbox_coords[i][j][0][1].tolist()), int(
                        bbox_coords[i][j][1][1].tolist())
                    label[i, s_bbx1:s_bbx2, s_bby1:s_bby2] = cut_patch[i, j, :, :]
    elif len(label.shape) == 4:
        for i in range(cut_patch.shape[0]):
            for j in range(cut_patch.shape[1]):
                for k in range(cut_patch.shape[2]):

                    if not (bbox_coords[i][j][0][0] == 0 and bbox_coords[i][j][1][0] == 0 and bbox_coords[i][j][1][0] == 0):
                        s_bbx1, s_bbx2 = int(bbox_coords[i][j][0][0].tolist()), int(
                            bbox_coords[i][j][1][0].tolist())
                        s_bby1, s_bby2 = int(bbox_coords[i][j][0][1].tolist()), int(
                            bbox_coords[i][j][1][1].tolist())
                        s_bbz1, s_bbz2 = int(bbox_coords[i][j][0][2].tolist()), int(
                            bbox_coords[i][j][1][2].tolist())
                        label[i, s_bbx1:s_bbx2,
                              s_bby1:s_bby2,s_bbz1:s_bbz2] = cut_patch[i, j, :, :,:]

    return label


def rand_bbox(image_size, cut_size, center_index=None):
    if len(image_size) == 2:
        W, H = image_size
        cut_w, cut_h = cut_size

        # uniform
        if center_index is None or len(center_index) == 0:
            cx = np.random.randint(cut_w//2, W-cut_w//2)
            cy = np.random.randint(cut_h//2, H-cut_h//2)
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
        
    elif len(image_size) == 3:
        W, H, S = image_size
        cut_w, cut_h, cut_s = cut_size

        # uniform
        if center_index is None or len(center_index) == 0:
            cx = np.random.randint(cut_w//2, W-cut_w//2)
            cy = np.random.randint(cut_h//2, H-cut_h//2)
            cz = np.random.randint(cut_s//2, S-cut_s//2)
        else:

            random_index = np.random.randint(len(center_index))
            cx, cy, cz = center_index[random_index]
            # cy = center_index[random_index][1]
        if cx - cut_w // 2 < 0:
            cx = cut_w // 2
        elif cx + cut_w // 2 > W:
            cx = W - cut_w // 2

        if cy - cut_h // 2 < 0:
            cy = cut_h // 2
        elif cy + cut_h // 2 > H:
            cy = H - cut_h // 2
        
        if cz - cut_s // 2 < 0:
            cz = cut_s // 2
        elif cz + cut_s // 2 > H:
            cz = H - cut_s // 2

        bbx1 = cx - cut_w // 2
        bby1 = cy - cut_h // 2
        bbz1 = cz - cut_s // 2
        bbx2 = cx + cut_w // 2
        bby2 = cy + cut_h // 2
        bbz2 = cz + cut_s // 2
       

        return bbx1, bby1, bbz1, bbx2, bby2,bbz2


def adjust_bbox(large_lower, large_uper, small_lower, small_uper):
    half_size = (small_uper - small_lower) // 2
    center = large_lower + (large_uper - large_lower) // 2
    new_large_lower = center-half_size
    new_large_uper = center+half_size

    return new_large_lower, new_large_uper


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image