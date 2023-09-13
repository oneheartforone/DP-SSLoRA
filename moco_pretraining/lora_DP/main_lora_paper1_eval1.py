# experiment 1

# In this code we will train datasets for paper

###################################################################
# In this code, we study lora_model.
# DP version.
# imagenet vs. chexpert pretraining
######################################################################

# DP fine-tuning, the version of fixed noise_multiplier
# Base main_lincls_3_resnet18_2.py

# 0. DPSGD
# 1. classifier(no bias)
# 2. Zero initialization
# 3. Adam
# 4. LoRA
# 5. augmentation false
# 6. lr 0.5 (temp)
# 7. lr_decay True
# 8. image_size 320
# 9. pretrain BN 128



#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys

# get root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add project root path to this running files
sys.path.append(BASE_DIR)
print(BASE_DIR)

import argparse
import builtins
import random
import shutil
import time
import warnings
import math
import datetime
import getpass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import itertools
from lora import (
    inject_trainable_lora_extended,
    extract_lora_ups_down,
)

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

import evaluator_for_multi as eval_tools
import meters as meters
import moco.aihc_utils.image_transform as image_transform
import moco.training_tools.plot_utils as pltu

import wandb
# prohibit wandb
# os.environ['WANDB_MODE'] = 'offline'  # 0.13 以上版本
# os.environ['WANDB_MODE'] = 'dryrun'   # 0.9.7 以下

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Chexpert-medical Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
# JBY: Decrease number of workers
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('--pretrained', default=True, type=bool,
                    help='signal to moco pretrained checkpoint')

# Stanford AIHC modification
parser.add_argument('--best-metric', dest='best_metric', type=str, default='acc@1',
                    help='metric to use for best model')
parser.add_argument('--semi-supervised', dest='semi_supervised', action='store_true',
                    help='allow the entire model to fine-tune')
parser.add_argument('--binary', dest='binary', action='store_true', help='change network to binary class')
parser.add_argument('--maintain_ratio', dest='maintain_ratio', default=True, help='Using square resize image.')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--cos-rate', default=4, type=float, metavar='CR',
                    help='Scaling factor for cos, higher the slower the decay')

# medical images setting
parser.add_argument('--optimizer', dest='optimizer', default='adam',
                    help='optimizer to use, chexpert=adam')
parser.add_argument('--aug-setting', default='chexpert',
                    choices=['moco_v1', 'moco_v2', 'chexpert'],
                    help='version of data augmentation to use')

# DP parameters
parser.add_argument('--gn', type=bool, default=False, help="When use GN pretrain model, change BN to GN in downstream model")

best_metrics = {'acc@1': {'func': 'topk_acc', 'format': ':6.3f', 'args': [1]}}
                # 'acc@5': {'func': 'topk_acc', 'format': ':6.2f', 'args': [5]},
                # 'auc': {'func': 'compute_auc_binary', 'format': ':6.2f', 'args': []}}
best_metric_val = 0




def main():
    import warnings
    warnings.simplefilter("ignore")

    checkpoint_folder = get_storage_folder("moco", f'model_and_datasets')
    with wandb.init(config=None, dir=checkpoint_folder):

        config = wandb.config

        args = parser.parse_args()
        args.lr_decay = config.lr_decay
        args.lr = 0.001
        args.img_size = 320
        args.crop = 320
        args.train_aug = config.aug
        args.model_init = config.model_init
        args.schedule = [30, 38]
        args.batch_size = 256
        args.lora_rank = int(config.lora_rank)

        # choose pretrain parameter
        args.pretrain_para = config.pretrain_para
        if args.pretrain_para == "imagenet":
            args.from_imagenet = True
            print("Use Imagenet pretrain parameters. ")
        else:
            args.from_imagenet = False
            if args.pretrain_para == "chexpert128":
                args.pretrained_path = r"../../results/Pretrain_in_chexpert/all_none_tuningbatch128/checkpoint_0019.pth.tar"
                print("Use CheXpert128 pretrain parameters. ")

        # Whether to use DP
        args.dp = config.dp

        # DP parameter
        args.max_grad_norm = 1.
        args.noise_multiplier = 1.
        args.max_physical_batch_size = 64

        # linear or full or Lora
        args.fine_tuning = config.fine_tuning

        # dataset
        args.dataset = config.dataset

        # dataset path
        if args.dataset == "RSNA":
            args.train_data = r"../../data\RSNA\Split\two_classification\train"
            args.val_data = r"../../data\RSNA\Split\two_classification\valid"
            args.test_data = r"../../data\RSNA\Split\two_classification\test"
            args.delta = 1e-4
        elif args.dataset == "covid08G":
            args.train_data = r"../../data\covid_Radiography_Dataset08G\yin\train"
            args.val_data = r"../../data\covid_Radiography_Dataset08G\yin\valid"
            args.test_data = r"../../data\covid_Radiography_Dataset08G\yin\test"
            args.delta = 1e-4
        elif args.dataset == "covid11G":
            args.train_data = r"../../data\covid_Chest_X-Ray_Images11G\Train"
            args.val_data = r"../../data\covid_Chest_X-Ray_Images11G\balanced_11G\valid"
            args.test_data = r"../../data\covid_Chest_X-Ray_Images11G\balanced_11G\Test"
            args.delta = 1e-5
        elif args.dataset == "covid08_multi":
            args.train_data = r"../../data\covid_Radiography_Dataset08G\Balanced_\train"
            args.val_data = r"../../data\covid_Radiography_Dataset08G\Balanced_\valid"
            args.test_data = r"../../data\covid_Radiography_Dataset08G\Balanced_\test"
            args.delta = 1e-5
        else:
            raise SystemExit("Can't find this dataset： ", args.dataset)

        print(args)
        # wandb.config.update(args)
        main_(args, checkpoint_folder)


def main_(args, checkpoint_folder):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args, checkpoint_folder)


def main_worker(args, checkpoint_folder):
    global best_metrics
    best_metric_val = 0

    # Data loading code
    traindir = args.train_data
    valdir = args.val_data
    testdir = args.test_data

    if args.aug_setting == 'moco_v2':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

        test_augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    elif args.aug_setting == 'chexpert':
        train_augmentation = image_transform.get_transform(args, training=args.train_aug)
        test_augmentation = image_transform.get_transform(args, training=False)

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(train_augmentation))

    # not use distributed training
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(test_augmentation)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose(test_augmentation)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # suppress printing if not master
    if args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=args.from_imagenet)

    args.classes = os.listdir(args.val_data)
    num_classes = len(os.listdir(args.val_data))  # assume in imagenet format, so length == num folders/classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if args.model_init == "zero":
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    else:
        nn.init.xavier_normal_(model.fc.weight)
        nn.init.constant_(model.fc.bias, 0.0)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if not args.from_imagenet:
            if os.path.isfile(args.pretrained_path):
                checkpoint = torch.load(args.pretrained_path, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                print("=> loaded pre-trained model '{}'".format(args.pretrained_path))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained_path))

    # lora change
    model = ModuleValidator.fix(model)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if name in ['fc.weight', 'fc.bias']:
            param.requires_grad = True
    model_lora_params, _ = inject_trainable_lora_extended(model, target_replace_module={"BasicBlock"}, r=args.lora_rank)
    model_fc = [model.fc.parameters()]
    params_to_optimize = [
        {"params": itertools.chain(*model_lora_params)},
        {"params": itertools.chain(*model_fc)},
    ]
    print("length of trainable parameters: ", len(list(filter(lambda p: p.requires_grad, model.parameters()))))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print(model)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_to_optimize, args.lr,
                                    momentum=args.momentum,)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_to_optimize, args.lr,
                                     betas=(0.9, 0.999), weight_decay=False)
    # use dp
    if args.dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=args.max_grad_norm,
            noise_multiplier=args.noise_multiplier,
        )
        print("Used DP training")
    else:
        print("Used non-DP training")
        pass

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']

            # TODO JBY: Handle resume for current metrics setup
            raise NotImplementedError('Resuming not supported yet!')

            for metric in bestr_metrics:
                best_metrics[metric][0] = checkpoint[f'best_metrics'][metric]
            if args.gpu is not None:
                for metric in best_metrics:
                    best_metrics[metric][0] = best_metrics[metric][0].to(args.gpu)

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    evaluator = eval_tools.Evaluator(model, criterion, best_metrics,\
                                     {'train': train_loader,\
                                      'valid': val_loader,\
                                      'test': test_loader}, args)

    cm, cr, auc, all_output_best, all_gt_best = [], [], [], [], []  # Confusion Matrix, classification Report
    # when get the best acc, then save this epoch, acc and auc.
    for i, epoch in enumerate(range(args.start_epoch, args.epochs)):
        if args.lr_decay:
            cur_lr = adjust_learning_rate(optimizer, epoch, args)
        else:
            cur_lr = args.lr
        wandb_dict = {"current lr": cur_lr, "epoch": epoch}

        # train for one epoch
        if args.dp:
            cur_train_metrics, cur_train_losses, epsilon = train(train_loader, model, criterion, optimizer, epoch, args, best_metrics, privacy_engine)
        else:
            epsilon = 0
            cur_train_metrics, cur_train_losses = train(train_loader, model, criterion, optimizer, epoch, args, best_metrics, None)

        wandb_dict.update({"Train ACC@1": cur_train_metrics["acc@1"].avg,
                           "Train losses": cur_train_losses.avg, "epsilon": epsilon})

        metric_meters_ev, valid_losses, best_metric_for_epoch_ev, _, _, auc_ev, _, _, _ = evaluator.evaluate('valid', epoch, epoch)
        wandb_dict.update({"Valid Acc": metric_meters_ev["acc@1"].avg, "Valid Auc": auc_ev})

        metric_meters_et, test_losses, best_metric_for_epoch_et, cm_, cr_, auc_et, best_auc, all_output_best_, all_gt_best_ = evaluator.evaluate('test', 0, epoch)  # 0, But we should technically not optimize for this
        wandb_dict.update({"Test Acc@1": metric_meters_et["acc@1"].avg, "Test Auc": auc_et}) #?

        # test acc and auc
        is_best = best_auc >= best_metric_val  # args.best_metric default "auc"
        best_metric_val = max(best_metric_val, best_auc)  # auc

        wandb_dict.update({"Best Test Acc@1": best_metric_for_epoch_et["acc@1"], "Best Test AUC": best_metric_val})

        if is_best:
            wandb.run.summary["all_best_accuracy"] = list(best_metric_for_epoch_et.keys())[0]
            wandb.run.summary["all_best_auc"] = best_metric_val
            cm, cr = cm_, cr_
            all_output_best, all_gt_best = all_output_best_, all_gt_best_

        if is_best or epoch == args.epochs-1:  # effect running speed
            save_checkpoint(checkpoint_folder, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_metrics': {metric: evaluator.metric_best_vals[metric] for metric in evaluator.metric_best_vals},
                'best_metric_val': best_metric_val,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        wandb.log(wandb_dict)

    # print some metrics
    print("Best test auc and the corresponding acc and epoch:")
    for m in list(best_metric_for_epoch_et.keys())[:1]:
        print('{} {:.3f} '.format(m, best_metric_for_epoch_et[m]), end=' ')
    print("micro_auc {:.3f}".format(best_metric_val), end='  ')
    print(list(best_metric_for_epoch_et.keys())[-1], int(best_metric_for_epoch_et[list(best_metric_for_epoch_et.keys())[-1]]))
    if int(len(args.classes)) > 2:
        pltu.plot_roc_curve_multiclass(checkpoint_folder, args.classes, all_output_best, all_gt_best)
    else:
        pltu.plot_roc_curve(checkpoint_folder, args.classes, all_output_best, all_gt_best, )
    pltu.plot_confusion_matrix(checkpoint_folder, args.classes, cm, )

    # save a concise output
    with open(str(checkpoint_folder) + r"\best_metric_for_testset.txt", "w") as file:
        for key in best_metric_for_epoch_et:
            file.write(str(key) + ':' + str(best_metric_for_epoch_et[key]) + '\n')
        file.write("auc: " + str(best_metric_val) + "\n")
        file.write("Confusion Matrix: \n")
        cm = str(cm.tolist())
        file.write(cm + "\n")
        file.write("Classification Report: \n")
        file.write(cr + "\n")
        file.write("output best: \n")
        file.write(str(all_output_best.tolist()) + "\n")
        file.write("target: \n")
        file.write(str(all_gt_best.tolist()) + "\n")


def train(train_loader, model, criterion, optimizer, epoch, args, best_metrics, privacy_engine):
    print(f'==> Training, epoch {epoch}')

    batch_time = meters.AverageMeter('Time', ':6.3f')
    data_time = meters.AverageMeter('Data', ':6.3f')
    losses = meters.AverageMeter('Loss', ':.4e')

    metric_meters = {metric: meters.AverageMeter(metric,
                                                 best_metrics[metric]['format'])\
                                                    for metric in best_metrics}
    list_meters = [metric_meters[m] for m in metric_meters]

    if privacy_engine is not None:
        l_lorder = int(len(train_loader) * args.batch_size / args.max_physical_batch_size)
    else:
        l_lorder = len(train_loader)

    progress = meters.ProgressMeter(
        l_lorder,
        [batch_time, data_time, losses, *list_meters],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """

    model.train()

    all_output = []
    all_gt = []

    end = time.time()
    with BatchMemoryManager(data_loader = train_loader,
                            max_physical_batch_size=args.max_physical_batch_size,
                            optimizer = optimizer
                            )as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            optimizer.zero_grad()

            if args.gpu is not None:
                images = images.cuda(args.gpu)
            target = target.cuda(args.gpu)
            all_gt.append(target.cpu().detach().numpy())

            # compute output
            output = model(images)
            all_output.append(output.cpu().detach().numpy())

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))

            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            for metric in best_metrics:
                eval_args = [output, target, *best_metrics[metric]['args']]
                metric_func = eval_tools.__dict__[best_metrics[metric]['func']]
                result = metric_func(*eval_args)

                metric_meters[metric].update(result, images.size(0))

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


    all_output = np.concatenate(all_output)
    all_gt = np.concatenate(all_gt)

    for metric in best_metrics:
        args_ = [all_output, all_gt, *best_metrics[metric]['args']]
        metric_func = eval_tools.__dict__[best_metrics[metric]['func']]  # create function (auc or acc)
        result = metric_func(*args_)

        metric_meters[metric].update(result, images.size(0))

    progress.display(i + 1, summary=True)
    if args.dp:
        epsilon = privacy_engine.get_epsilon(args.delta)
        print("Epoch: ", epoch, "Epsilon: ", epsilon)

    if args.dp:
        return metric_meters, losses, epsilon
    else:
        return metric_meters, losses


def save_checkpoint(checkpoint_folder, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint_folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_folder, filename),
                        os.path.join(checkpoint_folder, 'model_best.pth.tar'))


# JBY: Ported over support for Cosine learning rate
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        # TODO, JBY, is /4 an appropriate scale?
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs / args.cos_rate))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.2 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_storage_folder(exp_name, exp_type):

    try:
        jobid = os.environ["SLURM_JOB_ID"]
    except:
        jobid = None

    datestr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    username = str(getpass.getuser())

    fname = f'{exp_name}_{exp_type}_{datestr}_SLURM{jobid}' if jobid is not None else f'{exp_name}_{exp_type}_{datestr}'

    path_name = STORAGE_ROOT / fname
    os.makedirs(path_name)

    print(f'Experiment storage is at {path_name}')
    return path_name

# project_name = "paper_3_moco_LoRA_fixed_noisemultiplier_1"
project_name = "test"
sweep_configuration = {
    'method': 'grid',
    'metric': {
        'goal': 'maximize',
        'name': 'Test Auc',
    },

    'parameters': {
        # 'pretrain_para': {"values": ['imagenet', 'cheXpert']},  # choose parameters
        # 'dp': {'values': [True]},  # Whether to use DP

        # 'model_init': {"values": ["normal", "zero"]},
        'model_init': {"values": ["normal"]},
        'dataset': {'values': ["covid08G", "covid11G", "RSNA"]},  # choose dataset
        'lora_rank': {'values': [5, ]},
        'aug': {'values': [True]},  # train augment
        'lr_decay': {'values': [True]},  # lr decay
        'pretrain_para': {"values": ["chexpert128"]},  # choose parameters
        "fine_tuning": {'values': ['lora']},  # training methods
        'dp': {'values': [True, ]},  # Whether to use DP
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name, )
print('sweep_id：{}'.format(sweep_id))
STORAGE_ROOT = Path('../../results/' + project_name)

if __name__ == '__main__':
    wandb.agent(sweep_id, function=main, count=50)
    wandb.finish()



#################################################
# python main_lora_paper1_eval1.py -a resnet18 --epochs 50 --binary --workers 0 --gpu 0



