import glob
import os
import torch
import nibabel as nib
from torch.utils.data import Dataset

nib.imageglobals.logger.setLevel(40)


class Surf_TimeSeq_Single_Dataset(Dataset):
    def __init__(self, root_path, surf_name, input_dim=100):
        self.input_dim = input_dim
        self.surf_name = surf_name
        self.dtseries = self._get_dtseries_data(root_path)

    def __getitem__(self, index):
        return self.dtseries

    def __len__(self):
        return 5000

    def _get_dtseries_data(self, root_path):
        root_path = os.path.join(root_path, 'fmri')
        total_files = glob.glob(rf'{root_path}/*.dtseries.nii')
        dtseries_file = total_files[0]
        dtseries = nib.load(dtseries_file)
        seq_data = dtseries.dataobj[:, :]

        brain_axis = dtseries.header.get_axis(1)
        seq_data_align = surf_data_from_cifit(seq_data, brain_axis, self.surf_name)

        data = torch.from_numpy(seq_data_align).float()
        data_mean = torch.mean(data, dim=-1, keepdim=True)
        data_std = torch.std(data, dim=-1, keepdim=True) + 1e-8

        return (data - data_mean) / data_std


def surf_data_from_cifit(data, axis, surf_name):
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():
        if name == surf_name:
            data = data.T[data_indices]
            return data
    raise ValueError(f"No structure named {surf_name}")
