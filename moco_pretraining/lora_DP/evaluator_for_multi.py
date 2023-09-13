import argparse
import os
import random
import time
import warnings
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report

from meters import AverageMeter
from meters import ProgressMeter
from combiner import detach_tensor

def decorator_detach_tensor(function):
    def wrapper(*args, **kwargs):
        # TODO Find a simple way to handle this business ...
        # If is eval, or if fast debug, or
        # is train and not heavy, or is train and heavy
        output = detach_tensor(args[0])
        target = detach_tensor(args[1])
        args = args[2:]

        result = function(output, target, *args, **kwargs)
        return result

    return wrapper


@decorator_detach_tensor
def topk_acc(output, target, k):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    argsorted_out = np.argsort(output)[:, -k:]
    matching = np.asarray(np.any(argsorted_out.T == target, axis=0))
    return matching.mean(dtype='f')


@decorator_detach_tensor
def compute_auc_binary(output, target):
    # assuming output and target are all vectors for binary case
    try:
        o = softmax(output, axis=1)  # output
        o_ = np.argmax(o, axis=1)
        auc = roc_auc_score(target, o[:, 1])

        # Confusion Matrix
        cm = confusion_matrix(target, o_)
        # Classification Report
        cr = classification_report(target, o_)

    except:
        return -1
    return cm, cr, auc


@decorator_detach_tensor
def compute_auc_multi(output, target):
    encoder = OneHotEncoder(sparse=False)

    try:
        o = softmax(output, axis=1)  # output
        o_ = np.argmax(o, axis=1)

        # AUROC
        targets_onehot = encoder.fit_transform(np.array(target).reshape(-1, 1))
        fpr, tpr, _ = roc_curve(targets_onehot.ravel(), o.ravel(), )
        micro_auc = auc(fpr, tpr)

        # Confusion Matrix
        cm = confusion_matrix(target, o_)
        # Classification Report
        cr = classification_report(target, o_)

    except:
        return -1
    return cm, cr, micro_auc


class Evaluator:

    def __init__(self, model, loss_func, metrics, loaders, args):

        self.model = model
        self.loss_func = loss_func
        self.metrics = metrics
        self.loaders = loaders
        self.args = args

        self.best_auc = 0.
        self.all_output_best = []
        self.all_gt_best = []
        self.metric_best_vals = {metric: 0 for metric in self.metrics}
        self.metric_best_vals.update({"epoch": 0})

    def evaluate(self, eval_type, epoch, real_epoch):

        print(f'==> Evaluation for {eval_type}, epoch {real_epoch}')

        loader = self.loaders[eval_type]

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        metric_meters = {metric: AverageMeter(metric, self.metrics[metric]['format']) \
                         for metric in self.metrics}
        list_meters = [metric_meters[m] for m in metric_meters]

        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, *list_meters],
            prefix=f'{eval_type}@Epoch {epoch}: ')

        # switch to evaluate mode
        self.model.eval()
        all_output = []
        all_gt = []

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                if self.args.gpu is not None:
                    images = images.cuda(self.args.gpu, non_blocking=True)
                target = target.cuda(self.args.gpu, non_blocking=True)
                all_gt.append(target.cpu())

                # compute output
                output = self.model(images)
                all_output.append(output.cpu())

                loss = self.loss_func(output, target)

                # JBY: For simplicity do losses first
                losses.update(loss.item(), images.size(0))

                for metric in self.metrics:
                    args_ = [output, target, *self.metrics[metric]['args']]
                    metric_func = globals()[self.metrics[metric]['func']]
                    result = metric_func(*args_)

                    metric_meters[metric].update(result, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #    .format(top1=top1, top5=top5))
            progress.display(i + 1)

        all_output = np.concatenate(all_output)
        all_gt = np.concatenate(all_gt)

        # Confusion Matrix, classification Report
        # cm, cr, macro_auc, weighted_auc = compute_auc_multi(all_output, all_gt)
        if 2 == len(self.args.classes):
            cm, cr, auc = compute_auc_binary(all_output, all_gt)
        else:
            cm, cr, auc = compute_auc_multi(all_output, all_gt)

        for metric in self.metrics:
            args_ = [all_output, all_gt, *self.metrics[metric]['args']]
            metric_func = globals()[self.metrics[metric]['func']]
            result = metric_func(*args_)

            metric_meters[metric].update(result, images.size(0))


        if auc >= self.best_auc:
            # save best_acc while saving the corresponding auc and epoch
            self.metric_best_vals["acc@1"] = metric_meters["acc@1"].avg
            self.metric_best_vals["epoch"] = real_epoch
            self.best_auc = auc  # auc
            self.all_output_best, self.all_gt_best = all_output, all_gt

        progress.display(i + 1, summary=True)

        if len(self.args.classes) == 2:
            print("\tauc {}".format(auc))
        else:
            print("\tmicro_auc {}".format(auc))

        if eval_type == 'test':
            # print("\tmacro_auc: {}\n\tweighted_auc: {}".format(macro_auc, weighted_auc))
            print("Confusion Matrix:\n", cm)
            print("Classification Report:\n", cr)

        return metric_meters, losses, self.metric_best_vals, cm, cr, auc, self.best_auc, self.all_output_best, self.all_gt_best
