import math
import sys
from typing import Iterable
import torch.nn.functional as F

import numpy as np
import torch

from src.utils import misc, lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader_LR: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    t1_coordinate_LR,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(model, 'kl_loss'):
        metric_logger.add_meter('kl_loss', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    if hasattr(model, 'mean_loss'):
        metric_logger.add_meter('mean_loss', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    if hasattr(model, 're_loss'):
        metric_logger.add_meter('re_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if hasattr(model, 'u_loss'):
        metric_logger.add_meter('u_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 表示左右半脑训练
    if args.hem == 'L':
        index = 0
    elif args.hem == 'R':
        index = 1
    else:
        print("Please indicate the brain hem!")
        sys.exit(1)
    data_loader = data_loader_LR[index]
    t1_coordinate = t1_coordinate_LR[index]
    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        if samples.shape[-1] != args.n_input:
            print("WARNING: input dimensions do not match output dimensions!")
            continue
        with torch.cuda.amp.autocast():
            predict, loss = model(samples, t1_coordinate)
            # x_bar, _ = model(samples)
            # loss = F.mse_loss(samples, x_bar)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if hasattr(model, 'kl_loss'):
            metric_logger.update(kl_loss=model.kl_loss)
        if hasattr(model, 'mean_loss'):
            metric_logger.update(mean_loss=model.mean_loss)
        if hasattr(model, 're_loss'):
            metric_logger.update(re_loss=model.re_loss)
        if hasattr(model, 'u_loss'):
            metric_logger.update(u_loss=model.u_loss)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_10x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_10x = int((data_iter_step / len(data_loader) + epoch) * 10)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_10x)
            log_writer.add_scalar('lr', lr, epoch_10x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if hasattr(model, 'change_time'):
        train_stats['change_time'] = model.change_time

    return train_stats
