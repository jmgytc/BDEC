import numpy as np
import torch
from tqdm import tqdm


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    X_np = X.detach().cpu().numpy()
    initial_state = []
    for i in range(num_clusters):
        num_samples = len(X_np)
        indices = np.random.choice(num_samples, 1, replace=False)
        initial_state.append(X_np[indices].squeeze(axis=0))
        ids = np.where((X_np != X_np[indices].squeeze(axis=0)).any(axis=1))
        X_np = X_np[ids]

    return torch.from_numpy(np.array(initial_state))

def initialize_plus(X, num_clusters):
    """
    initialize cluster centers of kmeans++
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    k = 1
    random_int = torch.randint(0, len(X), (1,))[0]
    # [k, n_z]
    centroid = X[random_int:random_int+1]
    while k < num_clusters:
        # 计算所有顶点到目前已有质心的距离 [n, k]
        distance_sim = torch.sum((X.unsqueeze(dim=-2) - centroid.unsqueeze(-3)) ** 2, dim=-1,
                                 keepdim=False)
        # 每个顶点选取到质心的最短距离 [n]
        distance_min = torch.min(distance_sim, dim=-1)[0]
        # 选取最远距离的下标 1
        distance_max_index = torch.argmax(distance_min)
        centroid_temp = X[distance_max_index:distance_max_index+1]
        centroid = torch.concat((centroid, centroid_temp), dim=0)
        k += 1
    return centroid

def kmeans(
        X,
        num_clusters,
        axis_ids3d,
        distance='euclidean',
        tol=1e-4,
        device='cuda'
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster label, cluster centers, center ids
    """
    # print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError
    device = torch.device(device)
    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    axis_ids3d = axis_ids3d.to(device)

    # initialize
    # initial_state = initialize(X, num_clusters)
    with torch.no_grad():
        initial_state = initialize_plus(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        center_ids = []

        dis = pairwise_distance_function(X, initial_state, device)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol or iteration > 1000:
            return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmax(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1.0 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

