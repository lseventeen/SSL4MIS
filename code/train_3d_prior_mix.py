import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.LA_dataset import (LAHeart, TwoStreamBatchSampler,Pr)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_urpc_util import test_all_case
from dataloaders.prior_mix import label_mix_with_patch,PriorMixAugment


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    num_classes = args.num_classes
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model = model.cuda()

    

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_labeled_num))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       labeled_idxs=labeled_idxs,
                       transform=PriorMixAugment(args.patch_size,args.cut_rate,args.mix_prob,args.max_mix_num))
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    model.train()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            image, label, bbox_coords, mix_patchs = (
                sampled_batch["image"],
                sampled_batch["label"],
                sampled_batch["bbox_coords"],
                sampled_batch["mix_patchs"]
            )
            image, label, bbox_coords, mix_patchs = (
                image.cuda(),
                label.cuda(),
                bbox_coords.cuda(),
                mix_patchs.cuda()
            )

            outputs = model(image)
            outputs_soft = torch.softmax(outputs, dim=1)

            pseudo_outputs = torch.argmax(
                outputs_soft.detach(),
                dim=1,
                keepdim=False,
            )
            pseudo_outputs_mix = label_mix_with_patch(
                pseudo_outputs[args.labeled_bs:], mix_patchs[args.labeled_bs:], bbox_coords[args.labeled_bs:])

            consistency_weight = 0.5
            # supervised loss
            sup_loss = ce_loss(outputs[: args.labeled_bs], label[: args.labeled_bs].long(),) + dice_loss(
                outputs_soft[: args.labeled_bs], label[: args.labeled_bs].unsqueeze(
                    1),
            )

            unsup_loss = ce_loss(outputs[args.labeled_bs:], pseudo_outputs_mix.long()) + dice_loss(
                outputs_soft[args.labeled_bs:],
                pseudo_outputs_mix.unsqueeze(1),
            )
            loss = sup_loss + consistency_weight * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/consistency_weight",
                              consistency_weight, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/unsup_loss", unsup_loss, iter_num)
            writer.add_scalar("info/sup_loss", sup_loss, iter_num)
            logging.info("iteration %d : model loss : %f" %
                         (iter_num, loss.item()))

            if iter_num % 50 == 0:
                image = image[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = pseudo_outputs.unsqueeze(0)[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="test.list", num_classes=num_classes, patch_size=args.patch_size,
                    stride_xy=18, stride_z=4)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                for cls in range(1, num_classes):
                    writer.add_scalar('info/val_cls_{}_dice_score'.format(cls),
                                      avg_metric[cls - 1, 0], iter_num)
                    writer.add_scalar('info/val_cls_{}_hd95'.format(cls),
                                      avg_metric[cls - 1, 1], iter_num)
                writer.add_scalar('info/val_mean_dice_score',
                                  avg_metric[:, 0].mean(), iter_num)
                writer.add_scalar('info/val_mean_hd95',
                                  avg_metric[:, 1].mean(), iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (
                        iter_num, avg_metric[:, 0].mean(), avg_metric[:, 1].mean()))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/LA', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='GTV/uncertainty_rectified_pyramid_consistency', help='experiment_name')
    parser.add_argument('--model', type=str, default='VNet', help='model_name')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.1,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="output channel of network")
    # label and unlabel
    parser.add_argument('--labeled_bs', type=int, default=2,
                        help='labeled_batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=4,
                        help='labeled data')
    parser.add_argument('--total_labeled_num', type=int, default=80,
                        help='total labeled data')
    # costs
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str,
                        default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=400.0, help='consistency_rampup')
    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
