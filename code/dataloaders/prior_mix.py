import numpy as np


class prior_mix(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.#固定长度的正方形边长
    """

    def __init__(self, random_state=np.random.RandomState(0), prob=0.5, mix_num=1, cut_rate=0.25):
        self.random_state = random_state
        self.prob = prob
        self.mix_num = mix_num
        self.cut_rate = cut_rate

    def __call__(self, source_image, source_label, target_image, target_label, save_cut=False):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        if self.random_state.uniform() > self.prob:
            source_center_index = source_label[source_label != 0]
            target_center_index = target_label[target_label != 0]

            for n in range(self.mix_num):
                # 正方形区域中心点随机出现
                s_bbx1, s_bby1, s_bbx2, s_bby2 = self.rand_bbox_2d(
                    source_label.size(), self.cut_rate, source_center_index)  # 随机产生一个box的四个坐标
                t_bbx1, t_bby1, t_bbx2, t_bby2 = self.rand_bbox_2d(
                    target_label.size(), self.cut_rate, target_center_index)

                if s_bbx2 - s_bbx1 > t_bbx2 - t_bbx1:
                    s_bbx1, s_bbx2 = self.adjust_bbox(
                        s_bbx1, s_bbx2, t_bbx1, t_bbx2)
                elif s_bbx2 - s_bbx1 < t_bbx2 - t_bbx1:
                    t_bbx1, t_bbx2 = self.adjust_bbox(
                        t_bbx1, t_bbx2, s_bbx1, s_bbx2)

                if s_bby2 - s_bby1 > t_bby2 - t_bby1:
                    s_bby1, s_bby2 = self.adjust_bbox(
                        s_bby1, s_bby2, t_bby1, t_bby2)
                elif s_bby2 - s_bby1 < t_bby2 - t_bby1:
                    t_bby1, t_bby2 = self.adjust_bbox(
                        t_bby1, t_bby2, s_bby1, s_bby2)

                
                source_image[s_bbx1:s_bbx2,
                             s_bby1:s_bby2] = target_image[t_bbx1:t_bbx2, t_bby1:t_bby2]
                source_label[s_bbx1:s_bbx2,
                             s_bby1:s_bby2] = target_label[t_bbx1:t_bbx2, t_bby1:t_bby2]
        if save_cut:
            return source_image, source_label, (s_bbx1, s_bby1, s_bbx2, s_bby2), target_label[t_bbx1:t_bbx2, t_bby1:t_bby2]
        else:
            return source_image, source_label

    def rand_bbox_2d(size, cut_rate=0.25, center_index=None):
        W, H = size[0], size[1]

        cut_rat = np.sqrt(cut_rate)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        if center_index is None:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        else:
            random_index = np.random.randint(len(center_index))
            cx = center_index[random_index][0]
            cy = center_index[random_index][1]

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def adjust_bbox(large_lower, large_uper, small_lower, small_uper):
        half_size = (small_uper - small_lower) // 2
        center = large_lower + (large_uper - large_lower) // 2
        new_large_lower = center-half_size
        new_large_uper = center+half_size

        return new_large_lower, new_large_uper

    def rand_bbox_3d(size, size_rate):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(size_rate)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return {bbx1, bby1, bbx2, bby2}
