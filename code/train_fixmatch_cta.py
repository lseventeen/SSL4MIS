import argparse
import logging
import os
import re
import random
import shutil
import sys
import time
from xml.etree.ElementInclude import default_loader
# from more_itertools import sample

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributions import Categorical
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import augmentations
from PIL import Image

from dataloaders import utils
from dataloaders.dataset import (
    BaseDataSets,
    CTATransform,
    RandomGenerator,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume
from datetime import datetime
import wandb




def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
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


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def refresh_policies(db_train, cta):
        db_train.ops_weak = cta.policy(probe=False, weak=True)
        db_train.ops_strong = cta.policy(probe=False, weak=False)
        logging.info(f"\nWeak Policy: {db_train.ops_weak}")
        logging.info(f"Strong Policy: {db_train.ops_strong}")

    cta = augmentations.ctaugment.CTAugment()
    transform = CTATransform(args.patch_size, cta)

    # sample initial weak and strong augmentation policies (CTAugment)
    ops_weak = cta.policy(probe=False, weak=True)
    ops_strong = cta.policy(probe=False, weak=False)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transform,
        ops_weak=ops_weak,
        ops_strong=ops_strong,
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    model = create_model()
    ema_model = create_model(ema=True)
    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # if restoring previous models:
    if args.load:
        try:
            # check if there is previous progress to be restored:
            logging.info(f"Snapshot path: {snapshot_path}")
            iter_num = []
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename:
                    basename, extension = os.path.splitext(filename)
                    iter_num.append(int(basename.split("_")[2]))
            iter_num = max(iter_num)
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename and str(iter_num) in filename:
                    model_checkpoint = filename
        except Exception as e:
            logging.warning(f"Error finding previous checkpoints: {e}")

        try:
            logging.info(f"Restoring model checkpoint: {model_checkpoint}")
            model, optimizer, start_epoch, performance = util.load_checkpoint(
                snapshot_path + "/" + model_checkpoint, model, optimizer
            )
            logging.info(f"Models restored from iteration {iter_num}")
        except Exception as e:
            logging.warning(f"Unable to restore model checkpoint: {e}, using new model")

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    iter_num = int(iter_num)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        # track mean error for entire epoch
        epoch_errors = []
        # refresh augmentation policies with each new epoch
        refresh_policies(db_train, cta)

        for i_batch, sampled_batch in enumerate(trainloader):
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )

            # handle unfavorable cropping
            non_zero_ratio = torch.count_nonzero(label_batch) / (24 * 256 * 256)
            if non_zero_ratio <= 0.02:
                logging.info("Refreshing policy...")
                refresh_policies(db_train, cta)
                continue

            # model preds
            outputs_weak = model(weak_batch)
            outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
            outputs_strong = model(strong_batch)
            outputs_strong_soft = torch.softmax(outputs_strong, dim=1)

            # getting pseudo labels
            with torch.no_grad():
                outputs_soft = torch.softmax(model(weak_batch), dim=1)
                pseudo_outputs = torch.argmax(
                    outputs_soft.detach(),
                    dim=1,
                    keepdim=False,
                )

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # supervised loss (weak preds against ground truth)
            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )
            # unsupervised loss (strong preds against pseudo label)
            unsup_loss = ce_loss(outputs_strong[args.labeled_bs :], pseudo_outputs[args.labeled_bs :]) + dice_loss(
                outputs_strong_soft[args.labeled_bs :],
                pseudo_outputs[args.labeled_bs :].unsqueeze(1),
            )

            loss = sup_loss + consistency_weight * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1

            # track batch-level error, used to update augmentation policy
            epoch_errors.append(0.5 * loss.item())

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)

   
            logging.info("iteration %d : model loss : %f" % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                # show weakly augmented image
                image = weak_batch[1, 0:1, :, :]
                writer.add_image("train/Image", image, iter_num)
                # show strongly augmented image
                image_strong = strong_batch[1, 0:1, :, :]
                writer.add_image("train/StrongImage", image_strong, iter_num)
                # show model prediction (strong augment)
                outputs_strong = torch.argmax(outputs_strong_soft, dim=1, keepdim=True)
                writer.add_image("train/model_Prediction", outputs_strong[1, ...] * 50, iter_num)
                # show ground truth label
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)
                # show generated pseudo label
                pseudo_labs = pseudo_outputs[1, ...].unsqueeze(0) * 50
                writer.add_image("train/PseudoLabel", pseudo_labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
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
                        "model_iter_{}_dice_{}.pth".format(iter_num, round(best_performance, 4)),
                    )
                    save_best_path = os.path.join(snapshot_path, "{}_best_model.pth".format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info(
                    "iteration %d : model_mean_dice : %f model_mean_hd95 : %f" % (iter_num, performance, mean_hd95)
                )
            model.train()
            ema_model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, "model_iter_" + str(iter_num) + ".pth")
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

        # update policy parameter bins for sampling
        mean_epoch_error = np.mean(epoch_errors)
        cta.update_rates(db_train.ops_weak, 1.0 - 0.5 * mean_epoch_error)
        cta.update_rates(db_train.ops_strong, 1.0 - 0.5 * mean_epoch_error)
    writer.close()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../data/ACDC", help="Name of Experiment")
    parser.add_argument("--exp", type=str, default="FixMatch", help="experiment_name")
    parser.add_argument("--model", type=str, default="unet", help="model_name")
    parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
    parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
    parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
    parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
    parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    parser.add_argument("--num_classes", type=int, default=4, help="output channel of network")
    parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.8,
        help="confidence threshold for using pseudo-labels",
    )

    parser.add_argument("--labeled_bs", type=int, default=12, help="labeled_batch_size per gpu")
    # parser.add_argument('--labeled_num', type=int, default=136,
    parser.add_argument("--labeled_num", type=int, default=7, help="labeled data")
    # costs
    parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
    parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
    parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
    parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")
    parser.add_argument("-wm", "--wandb_mode",
                                required=False, default="online")
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

    snapshot_path = "../model/ACDC_{}_{}/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")
    shutil.copytree(".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"]))

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
               tags=["ss-net"], mode=args.wandb_mode,sync_tensorboard =True)
    train(args, snapshot_path)
