from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from src.models.AE import AE
from src.utils.kmeans import kmeans


# input : [batch_size, x, y ,z ,time]

class BDEC(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, batch_size, args, v=1.0, threshold=0.001):
        super(BDEC, self).__init__()

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.v = v
        self.n_z = n_z
        self.batch_size = batch_size
        self.args = args
        self.is_first = True
        self.n_clusters = n_clusters
        self.p = None

        self.cluster_layer = Parameter(torch.Tensor(batch_size, n_clusters, n_z))

        self.space_shape = None

        self.item = 0
        self.threshold = threshold
        self.change_time = 0
        self.change_rate_min = 1
        point_num = 29696 if args.hem == 'L' else 29716
        self.previous_predict_q = torch.zeros((batch_size, point_num), device=args.device)
        self.mean_q = torch.ones((batch_size, point_num, args.n_clusters), device=args.device) / args.n_clusters

        self.kl_loss = 0
        self.mean_loss = 0
        self.re_loss = 0
        self.u_loss = 0

        self.change_rate_queue = deque(maxlen=10)

        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, t1_coordinate):

        # x_bar : input
        # z : [batch_size,index,n_z]
        x_bar, z = self.ae(x)

        t1_coordinate = t1_coordinate / 100
        t1_coordinate = t1_coordinate.to(self.args.device)
        cos_multiple = torch.prod(t1_coordinate, dim=-1, keepdim=True) * 0.99 + 1

        if self.args.with_pos:
            z_with_pos = z * cos_multiple
        else:
            z_with_pos = z
        z_with_pos = (z_with_pos - z_with_pos.mean(dim=-1, keepdim=True)) / z_with_pos.std(dim=-1, keepdims=True)

        if self.training and not self.args.is_continue:
            if self.is_first:

                # [batch_size, num_cluster, n_z]
                cluster_centers_b = []
                for i in range(1):
                    _, centers = kmeans(
                        X=z_with_pos[0], num_clusters=self.n_clusters, axis_ids3d=t1_coordinate,
                        distance='euclidean', tol=0.000000001, device=self.args.device
                    )
                    cluster_centers_b.append(centers)
                cluster_centers_b = torch.stack(cluster_centers_b)
                cluster_centers_b = torch.mean(cluster_centers_b, dim=0, keepdim=True)

                self.cluster_layer.data = cluster_centers_b.to(self.args.device)

        # 基于距离的q分布
        cluster_u = self.cluster_layer
        distance_sim = torch.sum((z_with_pos.unsqueeze(dim=-2) - cluster_u.unsqueeze(-3)) ** 2, dim=-1,
                                 keepdim=False)

        q = 1.0 / (1e-5 + distance_sim / self.v)
        q = q.pow(-(self.v + 1.0) / 2.0)
        # q:[batch_size,index,cluster_num]
        q = q / torch.sum(q, dim=-1, keepdim=True)

        # predict:[batch_size,x,y,z]
        predict = torch.argmax(q, dim=-1, keepdim=False)
        clu_num = 0
        for i in range(len(predict)):
            clu, count = torch.unique(predict[i], return_counts=True)
            clu_num += len(clu)

        print(f'classify num：{int(clu_num / len(predict))}\n')

        change_rate = torch.sum((predict != self.previous_predict_q)) / self.previous_predict_q.numel()
        change_rate = change_rate.cpu()
        self.change_rate_queue.append(change_rate)
        change_rate_mean = np.mean(self.change_rate_queue)
        if self.change_rate_min > change_rate_mean and self.item > 5000:
            self.change_rate_min = change_rate_mean
        change_rate_min = self.change_rate_min

        if self.item % 1 == 0:

            c = self.args.c
            p, v, u = stat_target_distribution(q, c)
            if self.is_first:
                self.is_first = False
                self.p = p
            self.p = 0.1 * p + 0.9 * self.p.detach()
            p = self.p
        else:
            p = self.p.detach()

        q_log = q.log()
        kl_loss = F.kl_div(q_log, p, reduction='mean') * 0.01
        re_loss = F.mse_loss(x_bar, x)
        u_loss = cal_grad_loss(self.cluster_layer) * 0.01
        loss = kl_loss + re_loss + u_loss

        self.kl_loss = kl_loss
        self.re_loss = re_loss
        self.u_loss = u_loss

        print(f"kl_loss:{kl_loss.item()}\tre_loss:{re_loss}\tu_loss:{u_loss}\tLoss:{loss.item()}\t"
              f"change_rate_mean:{change_rate_mean}\tmin:{change_rate_min}")
        self.item += 1
        self.previous_predict_q = predict
        return predict, loss


# q:[batch_size,index,cluster_num]
def target_distribution(q):
    f = q.sum(dim=1, keepdim=True)
    weight = q ** 2 / f
    return weight / weight.sum(dim=-1, keepdim=True)


def stat_target_distribution(q, c):
    device = 'cuda'
    q = q.to(device)
    q_2 = q ** 2
    cluster_num = q.shape[-1]
    index = q.shape[1]
    batch_size = q.shape[0]
    u = q.sum(dim=1, keepdim=True)
    u = (u - u.mean(dim=-1, keepdims=True)) / u.std(dim=-1, keepdims=True)

    # N:[batch_size,cluster_num]
    N = torch.ones(batch_size, cluster_num, dtype=torch.long, device=device)
    predict = torch.argmax(q, dim=-1, keepdim=False)
    clu, count = torch.unique(predict, return_counts=True)
    if batch_size == 1:
        clu = clu.unsqueeze(dim=0)
        count = count.unsqueeze(dim=0)
    for i in range(batch_size):
        N[i, clu[i]] = count[i]
    # [batch_size,index*cluster_num]
    temp_q = (1 - q) ** 2 * torch.log(q) * index
    temp_q = torch.flatten(temp_q, start_dim=1)

    with torch.no_grad():
        v = []
        for i in range(cluster_num):
            v_i = torch.sqrt(-N[:, i:i + 1] / temp_q).sum(dim=1, keepdims=True)
            v.append(v_i)
        v = torch.concat(v, dim=-1).unsqueeze(dim=1)
        v = (v - v.mean(dim=-1, keepdims=True)) / v.std(dim=-1, keepdims=True)
    if batch_size == 1:
        u = u - u.min() + 1e-3
        v = v - v.min() + 1e-3
    else:
        u = u - u.min(dim=-1, keepdim=True).values + 1e-3
        v = v - v.min(dim=-1, keepdim=True).values + 1e-3

    f = u + v + c

    weight_1 = q_2 / f
    # [batch_size,index,1]
    weight_2 = weight_1.sum(dim=-1, keepdims=True)
    return ((weight_1 / weight_2).to('cuda'), v, u)


def cal_grad_loss(cluster_layer):
    # [batch_size, num_cluster, num_cluster]
    distance_sim = torch.sum((cluster_layer.unsqueeze(dim=-2) - cluster_layer.unsqueeze(-3)) ** 2, dim=-1,
                             keepdim=False)

    # [batch_size, num_cluster, num_cluster]
    mean_distance = distance_sim.sum() / (distance_sim.numel() - distance_sim.shape[0] * distance_sim.shape[1])
    grad_loss = 1 / mean_distance
    return grad_loss
