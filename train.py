import argparse
import os
import time

import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm

from common import config
from dataset import calc_data_weights, get_dataloaders
from model import (
                   resnet50, resnet101)
from utils import AUCMeter, AverageMeter, TrainClock, save_args
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = True
LOSS_WEIGHTS = calc_data_weights()


class Session:

    def __init__(self, config, net=None):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.net = net
        self.best_val_acc = 0.0
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        self.clock = TrainClock()

    def save_checkpoint(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        tmp = {
            "net": self.net.state_dict(),
            "best_val_acc": self.best_val_acc,
            "clock": self.clock.make_checkpoint(),
        }
        print("\n")
        print(f"Save ckpt to {ckp_path}...")
        torch.save(tmp, ckp_path)
        print("Done!")
        print("\n")

    def load_checkpoint(self, ckp_path):
        checkpoint = torch.load(ckp_path)
        self.net.load_state_dict(checkpoint["net"])
        self.clock.restore_checkpoint(checkpoint["clock"])
        self.best_val_acc = checkpoint["best_val_acc"]


def train_model(train_loader, model, optimizer, epoch):
    losses = AverageMeter("epoch_loss")
    accs = AverageMeter("epoch_acc")

    # ensure model is in train mode
    model.train()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        inputs = data["image"]
        labels = data["label"]
        study_type = data["meta_data"]["study_type"]
        file_paths = data["meta_data"]["file_path"]
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        weights = [
            LOSS_WEIGHTS[labels[i]][study_type[i]] for i in range(inputs.size(0))
        ]
        weights = torch.Tensor(weights).view_as(labels).to(config.device)

        outputs = model(inputs)
        preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)

        # update loss metric

        # Change [64] to [64,1]
        labels = labels.unsqueeze(1)
        weights = weights.unsqueeze(1)

        # loss = F.binary_cross_entropy(
        #     outputs, labels.to(config.device).float(), weights
        # )

        criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
        loss = criterion(outputs, labels.to(config.device).float())
        losses.update(loss.item(), inputs.size(0))

        corrects = torch.sum(preds.view_as(labels) == labels.float().data)
        acc = corrects.item() / inputs.size(0)
        accs.update(acc, inputs.size(0))

        # compute gradient and do SGD step
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, i, len(train_loader)))
        pbar.set_postfix(
            loss=":{:.4f}".format(losses.avg), acc=":{:.4f}".format(accs.avg)
        )

    outspects = {"epoch_loss": losses.avg, "epoch_acc": accs.avg}
    return outspects


# original validation function
def valid_model(valid_loader, model, optimizer, epoch):
    # using model to predict, based on dataloader
    losses = AverageMeter("epoch_loss")
    accs = AverageMeter("epoch_acc")
    model.eval()

    st_corrects = {st: 0 for st in config.study_type}
    nr_stype = {st: 0 for st in config.study_type}
    study_out = {}  # study level output
    study_label = {}  # study level label
    auc = AUCMeter()
    # auc = 0

    # evaluate the model
    pbar = tqdm(valid_loader)
    for k, data in enumerate(pbar):
        inputs = data["image"]
        labels = data["label"]
        encounter = data["meta_data"]["encounter"]
        study_type = data["meta_data"]["study_type"]
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        weights = [
            LOSS_WEIGHTS[labels[i]][study_type[i]] for i in range(inputs.size(0))
        ]
        weights = torch.Tensor(weights).view_as(labels).to(config.device)

        with torch.no_grad():

            outputs = model(inputs)
            preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)

            # Change [16] to [16,1]
            labels = labels.unsqueeze(1)
            weights = weights.unsqueeze(1)

            # update loss metric
            # loss = F.binary_cross_entropy(outputs, labels.to(config.device).float(), weights)
            criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
            loss = criterion(outputs, labels.to(config.device).float())
            losses.update(loss.item(), inputs.size(0))

            corrects = torch.sum(preds.view_as(labels) == labels.float().data)
            acc = corrects.item() / inputs.size(0)
            accs.update(acc, inputs.size(0))

            auc.add(preds, labels)

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, k, len(valid_loader)))
        pbar.set_postfix(
            loss=":{:.4f}".format(losses.avg),
            acc=":{:.4f}".format(accs.avg),
            auc=":{:.4f}".format(auc.value()[0]),
        )

        for i in range(len(outputs)):
            if study_out.get(encounter[i], -1) == -1:
                study_out[encounter[i]] = [outputs[i].item()]
                study_label[encounter[i]] = labels[i].item()
                nr_stype[study_type[i]] += 1
            else:
                study_out[encounter[i]] += [outputs[i].item()]

    # study level prediction
    study_preds = {
        x: (np.mean(study_out[x]) > 0.5) == study_label[x] for x in study_out.keys()
    }

    for x in study_out.keys():
        st_corrects[x[: x.find("_")]] += study_preds[x]

    # acc for each study type
    avg_corrects = {st: st_corrects[st] / nr_stype[st] for st in config.study_type}

    total_corrects = 0
    total_samples = 0

    for st in config.study_type:
        total_corrects += st_corrects[st]
        total_samples += nr_stype[st]

    # acc for the whole dataset
    total_acc = total_corrects / total_samples

    # auc value
    final_scores = [np.mean(study_out[x]) for x in study_out.keys()]
    auc_output = np.array(final_scores)
    auc_target = np.array(list(study_label.values()))
    auc.add(auc_output, auc_target)

    auc_val, tpr, fpr = auc.value()
    pbar.set_postfix(auc=":{:.4f}".format(auc_val))

    torch.cuda.empty_cache()

    outspects = {"epoch_loss": losses.avg, "epoch_acc": total_acc, "epoch_auc": auc_val}

    df = pd.DataFrame(
        {
            "epoch": [epoch],
            "ELBOW": [avg_corrects["ELBOW"]],
            "FINGER": [avg_corrects["FINGER"]],
            "FOREARM": [avg_corrects["FOREARM"]],
            "HAND": [avg_corrects["HAND"]],
            "HUMERUS": [avg_corrects["HUMERUS"]],
            "SHOULDER": [avg_corrects["SHOULDER"]],
            "WRIST": [avg_corrects["WRIST"]],
            "FEMUR": [avg_corrects["FEMUR"]],
            "LEG": [avg_corrects["LEG"]],
            "KNEE": [avg_corrects["KNEE"]],
            "epoch_loss": [outspects["epoch_loss"]],
            "epoch_acc": [outspects["epoch_acc"]],
            "epoch_auc": [outspects["epoch_auc"]],
        }
    )
    if os.path.isfile(config.acc_path):
        csv = pd.read_csv(config.acc_path)
        pd.concat([csv, df]).to_csv(config.acc_path, index=False)
    else:
        df.to_csv(config.acc_path, index=False)

    return outspects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="epoch number")

    parser.add_argument(
        "-b", "--batch_size", default=16, type=int, help="mini-batch size"
    )

    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=1e-4,
        type=float,
        help="initial learning rate",
    )

    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_path",
        # default="./output/lqn_mura_v3/model/best_model.pth.tar",
        type=str,
        required=False,
    )
    parser.add_argument("--exp_name", default=config.exp_name, type=str, required=False)

    parser.add_argument("--net", default="resnet50", type=str, required=False)

    parser.add_argument(
        "--local", default=True, action="store_true", help="train local branch"
    )
    args = parser.parse_args()
    print(args)

    config.exp_name = args.exp_name
    config.make_dir()
    save_args(args, config.log_dir)

    # get network
    if args.net == "resnet50":
        net = resnet50(pretrained=True, drop_rate=args.drop_rate)
    elif args.net == "resnet101":
        net = resnet101(pretrained=True, drop_rate=args.drop_rate)

    net = net.cuda()
    sess = Session(config, net=net)

    # get dataloader
    train_loader = get_dataloaders("train", batch_size=args.batch_size, shuffle=True)

    valid_loader = get_dataloaders("valid", batch_size=args.batch_size, shuffle=False)

    if args.continue_path and os.path.exists(args.continue_path):
        sess.load_checkpoint(args.continue_path)

    # start session
    clock = sess.clock
    tb_writer = sess.tb_writer
    sess.save_checkpoint("start.pth.tar")

    optimizer = optim.Adam(sess.net.parameters(), args.lr)

    scheduler = ReduceLROnPlateau(
        optimizer, "max", factor=0.1, patience=10, verbose=True
    )

    # start training
    for e in range(clock.epoch, args.epochs):
        train_out = train_model(
            train_loader, sess.net, optimizer, clock.epoch
        )
        valid_out = valid_model(
            valid_loader, sess.net, optimizer, clock.epoch
        )

        tb_writer.add_scalars(
            "loss",
            {"train": train_out["epoch_loss"], "valid": valid_out["epoch_loss"]},
            clock.epoch,
        )

        tb_writer.add_scalars(
            "acc",
            {"train": train_out["epoch_acc"], "valid": valid_out["epoch_acc"]},
            clock.epoch,
        )

        tb_writer.add_scalar("auc", valid_out["epoch_auc"], clock.epoch)

        tb_writer.add_scalar(
            "learning_rate", optimizer.param_groups[-1]["lr"], clock.epoch
        )
        scheduler.step(valid_out["epoch_auc"])

        if valid_out["epoch_auc"] > sess.best_val_acc:
            sess.best_val_acc = valid_out["epoch_auc"]
            sess.save_checkpoint("best_model.pth.tar")

        # Save after each 5 epoch.
        sess.save_checkpoint("epoch{}.pth.tar".format(clock.epoch))

        # sess.save_checkpoint("latest.pth.tar")

        clock.tock()


if __name__ == "__main__":
    main()
