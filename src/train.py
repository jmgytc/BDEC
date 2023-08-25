import json
import os
import sys

sys.path.append(os.path.abspath('Your Code path'))
import time
import datetime
from pathlib import Path
from src.models.BDEC import BDEC
cwd = os.getcwd()
print(cwd)
import numpy as np
import torch
import argparse
import nibabel as nib
from timm.optim import optim_factory
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from src.dataset.load_nii import Surf_TimeSeq_Single_Dataset
from src.utils import misc
from src.utils.engine_pretrain import train_one_epoch
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    hem = 'L'
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--root_path', type=str, default=fr'./data')

    parser.add_argument('--resume', default=fr'./weight/checkpoint-{hem}.pth',
                        help='resume from checkpoint')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--hem', type=str, default=f'{hem}')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_input', type=int, default=1200)
    parser.add_argument('--n_z', type=int, default=10)
    parser.add_argument('--n_clusters', type=int, default=200)
    parser.add_argument('--with_pos', type=bool, default=True)
    parser.add_argument('--c', type=int, default=10,
                        help='用于限制u+v的')
    parser.add_argument('--is_continue', type=bool, default=False,
                        help='whether to continue training')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    return parser

def _get_t1_coordinate_LR(root_path, template):
    vtx_indices = []
    dtseries = nib.load(template)
    brain_axis = dtseries.header.get_axis(1)
    assert isinstance(brain_axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in brain_axis.iter_structures():
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            vtx_indices.append(model.vertex)
        if name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            vtx_indices.append(model.vertex)
    t1_coordinate_LR = []
    root_path = os.path.join(root_path, 't1')
    for i, file in enumerate(os.listdir(root_path)):
        t1_coordinate = nib.load(os.path.join(root_path, file))
        t1_coordinate = torch.from_numpy(t1_coordinate.darrays[0].data[vtx_indices[i]]).float()
        t1_coordinate_LR.append(t1_coordinate)
    return t1_coordinate_LR


def main(args):
    misc.init_distributed_mode(args)
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    train_dataset_L = Surf_TimeSeq_Single_Dataset(root_path=args.root_path, surf_name='CIFTI_STRUCTURE_CORTEX_LEFT',
                                                  input_dim=args.n_input)
    train_loader_L = DataLoader(dataset=train_dataset_L, batch_size=args.batch_size, drop_last=True, num_workers=2)
    train_dataset_R = Surf_TimeSeq_Single_Dataset(root_path=args.root_path, surf_name='CIFTI_STRUCTURE_CORTEX_RIGHT',
                                                  input_dim=args.n_input)
    train_loader_R = DataLoader(dataset=train_dataset_R, batch_size=args.batch_size, drop_last=True)
    train_loader_LR = [train_loader_L, train_loader_R]
    t1_coordinate_LR = _get_t1_coordinate_LR(args.root_path, './data/fmri/average_data.dtseries.nii')

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # network
    model = BDEC(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 batch_size=args.batch_size,
                 args=args,
                 v=1.0).to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    with open(os.path.join(args.log_dir, 'architecture.txt'), 'w+') as f:
        args_str = "{}".format(args).replace(', ', ',\n') + '\n\n'
        model_str = "Model = %s" % str(model_without_ddp)
        f.write(args_str)
        f.write(model_str)

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    print(optimizer)
    loss_scaler = NativeScaler()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, train_loader_LR,
            optimizer, device, epoch, loss_scaler,
            t1_coordinate_LR,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
