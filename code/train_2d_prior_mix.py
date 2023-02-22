import argparse
import logging
import os
import re
import random
import shutil
import sys
import time
from xml.etree.ElementInclude import default_loader
from datetime import datetime
import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume
from dataloaders.prior_mix import label_mix_with_patch, PriorMixAugment

from dataloaders.dataset import (
    BaseDataSets,
 

    TwoStreamBatchSampler,
)


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "1": 32,
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "70": 1312,
        }
    elif "Prostate":
        ref_dict = {
            "2": 27,
            "4": 53,
            "8": 120,
            "12": 179,
            "16": 256,
            "21": 312,
            "42": 623,
        }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    total_slices = 1312
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        labeled_idxs=labeled_idxs,
        transform=PriorMixAugment(args.patch_size,args.cut_rate,args.mix_prob,args.max_mix_num))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    model = create_model()
    # create model for ema (this model produces pseudo-labels)
    # ema_model = create_model(ema=True)
    
    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    # set to train
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

            # outputs for model

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
            # loss = sup_loss + unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

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
                
                writer.add_image("train/Image",image[1, 0:1, :, :], iter_num)

                writer.add_image(
                    "train/Prediction", pseudo_outputs[1, ...].unsqueeze(0) * 50, iter_num)

                labs = label[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                # ema_model.eval()
                metric_list = 0.0
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume(
                            sampled_batch["image"],
                            sampled_batch["label"],
                            model,
                            classes=num_classes,
                        )
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/val_{}_hd95".format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num,
                    )

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance, 4)),
                    )
                    save_best_path = os.path.join(
                        snapshot_path, "{}_best_model.pth".format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info(
                    "iteration %d : model_mean_dice : %f model_mean_hd95 : %f" % (
                        iter_num, performance, mean_hd95)
                )
                model.train()
                # ema_model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str,
                        default="../data/ACDC", help="Name of Experiment")
    parser.add_argument("--exp", type=str,
                        default="baseline", help="experiment_name")
    parser.add_argument("--model", type=str, default="unet", help="model_name")
    parser.add_argument("--max_iterations", type=int,
                        default=30000, help="maximum epoch number to train")
    parser.add_argument("--batch_size", type=int,
                        default=24, help="batch_size per gpu")
    parser.add_argument("--deterministic", type=int, default=1,
                        help="whether use deterministic training")
    parser.add_argument("--base_lr", type=float, default=0.01,
                        help="segmentation network learning rate")
    parser.add_argument("--patch_size", type=list,
                        default=[256, 256], help="patch size of network input")
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="output channel of network")
    parser.add_argument("--load", default=False,
                        action="store_true", help="restore previous checkpoint")

    parser.add_argument("--labeled_bs", type=int, default=12,
                        help="labeled_batch_size per gpu")
    # parser.add_argument('--labeled_num', type=int, default=136,
    parser.add_argument("--labeled_num", type=int,
                        default=7, help="labeled data")
    # costs
    parser.add_argument("--mix_prob", type=float,
                        default=0.2, help="mix probability")
    parser.add_argument("--cut_rate", type=float,
                        default=0.2, help="cut rate")
    parser.add_argument("--max_mix_num", type=float,
                        default=1, help="cut rate")
    parser.add_argument("--wandb_mode",
                       type=str, default="offline")
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

    snapshot_path = "../model/ACDC/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")
    if args.exp != "debug":
        shutil.copytree(".", snapshot_path + "/code",
                        shutil.ignore_patterns([".git", "__pycache__", "wandb", ".vscode"]))

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    experiment_id = f"{args.exp}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    wandb.init(project=f"ssl 2D label {args.labeled_num}", name=experiment_id,
               tags=["baseline"], mode=args.wandb_mode, sync_tensorboard=True)

    train(args, snapshot_path)
