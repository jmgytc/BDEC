import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class AE_LayerNormal(AE):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE_LayerNormal, self).__init__(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                                             n_input, n_z)
        self.enc_layer_normal_1 = nn.LayerNorm(n_enc_1)
        self.enc_layer_normal_2 = nn.LayerNorm(n_enc_2)
        self.enc_layer_normal_3 = nn.LayerNorm(n_enc_3)

        self.dec_layer_normal_1 = nn.LayerNorm(n_dec_1)
        self.dec_layer_normal_2 = nn.LayerNorm(n_dec_2)
        self.dec_layer_normal_3 = nn.LayerNorm(n_dec_3)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_layer_normal_1(self.enc_1(x)))
        enc_h2 = F.relu(self.enc_layer_normal_2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.enc_layer_normal_3(self.enc_3(enc_h2)))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_layer_normal_1(self.dec_1(z)))
        dec_h2 = F.relu(self.dec_layer_normal_2(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.dec_layer_normal_3(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


def count_target_distribution(q):
    cluster_num = q.shape[-1]
    index = q.shape[1]
    batch_size = q.shape[0]
    # u: [batch_size,1,cluster_num]
    u = q.sum(dim=1, keepdim=True)

    # N:[batch_size,cluster_num]
    N = torch.zeros(batch_size, cluster_num, dtype=torch.long, device='cuda')
    predict = torch.argmax(q, dim=-1, keepdim=False)
    clu, count = torch.unique(predict, return_counts=True)
    if batch_size == 1:
        clu = clu.unsqueeze(dim=0)
        count = count.unsqueeze(dim=0)
    for i in range(batch_size):
        N[i, clu[i]] = count[i]
    N = ((N + 1) / index).unsqueeze(dim=1)

    f = q.sum(dim=1, keepdim=True)
    weight = q ** 2 / (f / N)
    return weight / weight.sum(dim=-1, keepdim=True)
