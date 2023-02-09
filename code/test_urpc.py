import argparse
import os
import shutil
from glob import glob
import numpy

import torch

from networks.unet_3D_dv_semi import unet_3D_dv_semi
from networks.unet_3D import unet_3D
from test_urpc_util import test_all_case


def net_factory(net_type="unet_3D", num_classes=3, in_channels=1):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=num_classes, in_channels=in_channels).cuda()
    elif net_type == "unet_3D_dv_semi":
        net = unet_3D_dv_semi(n_classes=num_classes,
                              in_channels=in_channels).cuda()
    else:
        net = None
    return net


def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    print(test_save_path)
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    net = net_factory(FLAGS.model, num_classes, in_channels=1)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, image_list,   num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                    default='../data/LA', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                    default='LA/Uncertainty_Rectified_Pyramid_Consistency_4', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default="unet_urpc", help='model_name')
    FLAGS = parser.parse_args()

    metric = Inference(FLAGS)
    print(metric)
