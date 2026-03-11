# -*- coding: utf-8 -*-
"""Training script for the DenseUAV two-view geo-localization model.

This script trains a dual-branch model that jointly learns satellite-view and
drone-view image representations for UAV geo-localization.  Configuration is
supplied via CLI flags that are then merged with an ``opts.yaml`` file found in
the same directory.

Example:
    Train with default settings on a single GPU::

        python train.py --name my_run --data_dir /path/to/DenseUAV/train \\
            --backbone ViTS-224 --head GeM --num_epochs 120 --batchsize 8

Outputs:
    - ``checkpoints/<name>/train.log``     — per-epoch training log.
    - ``checkpoints/<name>/net_<epoch>.pth`` — model weights saved every 10
      epochs starting from epoch 110.
    - ``checkpoints/<name>/opts.yaml``     — serialised configuration snapshot.
    - A copy of all source files under ``checkpoints/<name>/``.
"""

from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
# from models.model import make_model
from models.taskflow import make_model
from datasets.make_dataloader import make_dataset
from tool.utils import save_network, copyfiles2checkpoints, get_preds, get_logger, calc_flops_params, set_seed
import warnings
from losses.cal_loss import cal_loss
from losses.loss import Loss


warnings.filterwarnings("ignore")


def get_parse():
    """Parse command-line arguments and return the configuration namespace.

    Defines all training hyper-parameters as CLI flags.  After parsing, the
    resulting ``argparse.Namespace`` object is printed to stdout so that every
    run is self-documenting.

    Returns:
        argparse.Namespace: Parsed options including fields such as
            ``gpu_ids``, ``name``, ``data_dir``, ``batchsize``, ``lr``,
            ``backbone``, ``head``, ``num_epochs``, and various augmentation
            flags.
    """
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='test',
                        type=str, help='output model name')
    parser.add_argument('--data_dir', default='/Users/chibangnguyen/ayai/UAV/denseUAV_baseline/data/DenseUAV_data/train',
                        type=str, help='training dir path')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=256, type=int, help='height')
    parser.add_argument('--w', default=256, type=int, help='width')
    parser.add_argument('--rr', default="", type=str, help='random rotate')
    parser.add_argument('--ra', default="", type=str, help='random affine')
    parser.add_argument('--re', default="", type=str, help='random erasing')
    parser.add_argument('--cj', default="", type=str, help='color jitter')
    parser.add_argument('--erasing_p', default=0.3, type=float,
                        help='Random Erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=0, type=int,
                        help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--droprate', default=0.5,
                        type=float, help='drop rate')
    parser.add_argument('--DA', action='store_true',
                        help='use Color Data Augmentation')
    parser.add_argument('--share', action='store_true',
                        default=True, help='share weight between different view')
    parser.add_argument('--autocast', action='store_true',
                        default=True, help='use mix precision')
    parser.add_argument('--block', default=2, type=int, help='')
    parser.add_argument('--cls_loss', default="FocalLoss", type=str, help='')
    parser.add_argument('--feature_loss', default="WeightedSoftTripletLoss", type=str, help='')
    parser.add_argument('--kl_loss', default="", type=str, help='')
    parser.add_argument('--sample_num', default=1, type=int,
                        help='num of repeat sampling')
    parser.add_argument('--num_epochs', default=120, type=int, help='')
    parser.add_argument('--num_bottleneck', default=512, type=int, help='')
    parser.add_argument('--load_from', default="", type=str, help='')
    parser.add_argument('--backbone', default="ViTS-224", type=str, help='')
    parser.add_argument('--head', default="GeM", type=str, help='')
    # new: configurable checkpoint root for Modal Volume mount
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str,
                        help='root directory for checkpoints and logs (default: ./checkpoints)')

    opt = parser.parse_args()
    print(opt)
    return opt


def train_model(model, opt, optimizer, scheduler, dataloaders, dataset_sizes):
    """Run the full training loop for the dual-branch geo-localization model.

    Iterates over ``opt.num_epochs`` epochs.  Each epoch:

    1. Iterates over paired satellite / drone mini-batches.
    2. Performs a forward pass under ``torch.cuda.amp.autocast`` (mixed
       precision) when ``opt.autocast`` is ``True``.
    3. Computes composite loss: classification loss + triplet loss + KL loss.
    4. Back-propagates with ``GradScaler`` (AMP) or plain ``loss.backward()``.
    5. Logs per-epoch statistics to both file and stdout.
    6. Saves a checkpoint every 10 epochs starting from epoch 110.

    Args:
        model (torch.nn.Module): The dual-branch model to be trained.
        opt (argparse.Namespace): Configuration namespace produced by
            :func:`get_parse`.  Must contain at least ``use_gpu``,
            ``num_epochs``, ``name``, ``batchsize``, and ``autocast``.
        optimizer (torch.optim.Optimizer): Optimiser instance (e.g. AdamW or
            SGD) returned by ``make_optimizer``.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning-rate
            scheduler stepped once per epoch.
        dataloaders (torch.utils.data.DataLoader): Paired dataloader that
            yields ``(data_satellite, data_drone)`` tuples each iteration.
        dataset_sizes (dict): Mapping from split name to sample count, e.g.
            ``{'satellite': 54000, 'drone': 54000}``.  Used to normalise
            running statistics.
    """
    # new: use opt.checkpoint_dir so Modal can write to Volume
    checkpoint_root = getattr(opt, 'checkpoint_dir', 'checkpoints')
    log_path = os.path.join(checkpoint_root, opt.name, 'train.log')
    logger = get_logger(log_path)

    # thop MACs computation (commented out for production runs)
    # macs, params = calc_flops_params(
    #     model, (1, 3, opt.h, opt.w), (1, 3, opt.h, opt.w))
    # logger.info("model MACs={}, Params={}".format(macs, params))

    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs
    since = time.time()
    scaler = GradScaler()
    nnloss = Loss(opt)
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 50)

        model.train(True)  # Set model to training mode
        running_cls_loss = 0.0
        running_triplet = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0
        for data, data3 in dataloaders:
            # Retrieve drone and satellite input batches
            inputs, labels = data
            inputs3, labels3 = data3
            now_batch_size = inputs.shape[0]
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue
            if use_gpu:
                inputs = Variable(inputs.cuda().detach())
                inputs3 = Variable(inputs3.cuda().detach())
                labels = Variable(labels.cuda().detach())
                labels3 = Variable(labels3.cuda().detach())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # Zero gradients before forward pass
            optimizer.zero_grad()

            # Forward pass with optional mixed-precision context
            with autocast():
                outputs, outputs2 = model(inputs, inputs3)

            # Compute composite loss
            loss, cls_loss, f_triplet_loss, kl_loss = nnloss(
                outputs, outputs2, labels, labels3)

            # Backward pass
            if opt.autocast:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Accumulate loss statistics
            running_loss += loss.item() * now_batch_size
            running_cls_loss += cls_loss.item()*now_batch_size
            running_triplet += f_triplet_loss.item() * now_batch_size
            running_kl_loss += kl_loss.item() * now_batch_size

            # Accumulate accuracy statistics
            preds, preds2 = get_preds(outputs[0], outputs2[0])
            if isinstance(preds, list) and isinstance(preds2, list):
                running_corrects += sum([float(torch.sum(pred == labels.data))
                                        for pred in preds])/len(preds)
                running_corrects2 += sum([float(torch.sum(pred == labels3.data))
                                         for pred in preds2]) / len(preds2)
            else:
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects2 += float(torch.sum(preds2 == labels3.data))

        # Compute epoch-level averages
        epoch_cls_loss = running_cls_loss/dataset_sizes['satellite']
        epoch_kl_loss = running_kl_loss / dataset_sizes['satellite']
        epoch_triplet_loss = running_triplet/dataset_sizes['satellite']
        epoch_loss = running_loss / dataset_sizes['satellite']
        epoch_acc = running_corrects / dataset_sizes['satellite']
        epoch_acc2 = running_corrects2 / dataset_sizes['satellite']

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
        lr_other = optimizer.state_dict()['param_groups'][1]['lr']
        logger.info('Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                    .format(epoch_loss, epoch_cls_loss, epoch_kl_loss,
                            epoch_triplet_loss, epoch_acc,
                            epoch_acc2, lr_backbone, lr_other))

        scheduler.step()
        # old: chỉ lưu từ epoch >= 110
        # if epoch % 10 == 9 and epoch >= 110:
        #     save_network(model, opt.name, epoch, checkpoint_root=checkpoint_root)
        # old (mới hơn): lưu mỗi 10 epoch (9, 19, 29, ...)
        # if epoch % 10 == 9:
        #     save_network(model, opt.name, epoch, checkpoint_root=checkpoint_root)
        # new: lưu checkpoint mỗi epoch để có thể resume chi tiết nhất
        save_network(model, opt.name, epoch, checkpoint_root=checkpoint_root)

        time_elapsed = time.time() - since
        since = time.time()
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    set_seed(666)

    opt = get_parse()
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    # set gpu ids (only when CUDA is available to avoid CPU-only PyTorch error; safe on Modal)
    if use_gpu and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    dataloaders, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)

    model = make_model(opt)

    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)

    if use_gpu:
        model = model.cuda()
    # Copy source files to checkpoint directory for reproducibility
    copyfiles2checkpoints(opt)

    train_model(model, opt, optimizer_ft, exp_lr_scheduler,
                dataloaders, dataset_sizes)
