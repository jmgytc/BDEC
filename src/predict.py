import numpy as np
import torch
import argparse
import nibabel as nib

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from src.dataset.load_nii import Surf_TimeSeq_Single_Dataset
from src.models.BDEC import BDEC
from src.train import _get_t1_coordinate_LR
from src.utils import misc

root_path = f'./data'
dir_name = 'default'
hem = 'L'
label_name = f'label.txt'


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--root_path', type=str, default=f'{root_path}')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_input', type=int, default=1200)
    parser.add_argument('--n_z', type=int, default=10)
    parser.add_argument('--with_pos', type=bool, default=True)
    parser.add_argument('--c', type=int, default=10)
    parser.add_argument('--n_clusters', type=int, default=200)
    parser.add_argument('--resume', default=fr'./weight/checkpoint-{hem}.pth',
                        help='resume from checkpoint')
    parser.add_argument('--template', default=r'./data/fmri/average_data.dtseries.nii')
    parser.add_argument('--output_dir', default=fr'./output/{dir_name}',
                        help='path where to save, empty for no saving')
    parser.add_argument('--hem', type=str, default=f'{hem}')
    return parser


def main(args):
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cuda")
    cudnn.benchmark = True

    predict_dataset_L = Surf_TimeSeq_Single_Dataset(root_path=args.root_path, surf_name='CIFTI_STRUCTURE_CORTEX_LEFT',
                                                    input_dim=args.n_input)
    predict_loader_L = DataLoader(dataset=predict_dataset_L, batch_size=args.batch_size, drop_last=True)
    predict_dataset_R = Surf_TimeSeq_Single_Dataset(root_path=args.root_path, surf_name='CIFTI_STRUCTURE_CORTEX_RIGHT',
                                                    input_dim=args.n_input)
    predict_loader_R = DataLoader(dataset=predict_dataset_R, batch_size=args.batch_size, drop_last=True)

    predict_loader_LR = [predict_loader_L, predict_loader_R]

    t1_coordinate_LR = _get_t1_coordinate_LR(args.root_path, args.template)

    if args.hem == 'L':
        surf_name = 'CIFTI_STRUCTURE_CORTEX_LEFT'
    elif args.hem == 'R':
        surf_name = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    else:
        assert 'No hem'
    brain_total_zero, vtx_indices = surf_index_from_cifit(args.template, surf_name)

    # network
    model_without_ddp = BDEC(500, 500, 2000, 2000, 500, 500,
                             n_input=args.n_input,
                             n_z=args.n_z,
                             n_clusters=args.n_clusters,
                             batch_size=args.batch_size,
                             args=args,
                             v=1.0).to(device)

    load_model(args, model_without_ddp)
    model = model_without_ddp
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(1)
    print_freq = 1

    if args.hem == 'L':
        index = 0
    elif args.hem == 'R':
        index = 1

    data_loader = predict_loader_LR[index]
    t1_coordinate = t1_coordinate_LR[index]

    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.cuda.empty_cache()
        samples = samples.to(device)
        if samples.shape[-1] != args.n_input:
            print("WARNING: input dimensions do not match output dimensions!")
            continue
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                predict, _ = model(samples, t1_coordinate)
        predict += 1
        predict = predict.cpu().detach().numpy()
        brain_total_zero[vtx_indices] = predict[0]
        save2txt(brain_total_zero, args.output_dir + f'/{label_name}')
        break


def load_model(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        weight_dict = checkpoint['model']
        original_dict = model_without_ddp.state_dict()

        for original_key, original_value in original_dict.items():
            for weight_key, weight_value in weight_dict.items():
                if weight_key in original_key:
                    if weight_value.size() == original_value.size():
                        print(f"{original_key} shape is same")
                        original_dict[original_key] = weight_dict[weight_key]
        model_without_ddp.load_state_dict(original_dict)
        print("Resume checkpoint %s" % args.resume)


def surf_index_from_cifit(template_path, surf_name):
    data = nib.load(template_path)
    axis = data.header.get_axis(1)
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():
        if name == surf_name:
            vtx_indices = model.vertex
            surf_data = np.zeros(vtx_indices.max() + 1, dtype=int)
            return surf_data, vtx_indices
    raise ValueError(f"No structure named {surf_name}")


def save2txt(q, filename):
    file_handle = open(filename, mode='x')

    q = q.tolist()

    strNums = [str(q_i) for q_i in q]
    str1 = "\n".join(strNums)

    file_handle.write(str1)
    print(f'write {filename} success！')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
