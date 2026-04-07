import numpy as np
import torch
import pandas as pd
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """
    计算余弦距离矩阵。
    :param x1: 第一个输入张量 (n, d)
    :param x2: 第二个输入张量 (m, d)，如果为 None，则与 x1 相同
    :param eps: 防止除零的小常数
    :return: 余弦距离矩阵 (n, m)
    """
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)  # 计算 x1 的 L2 范数
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)  # 计算 x2 的 L2 范数
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)  # 计算余弦距离


def to_sparse(x):
    """
    将稠密张量转换为稀疏张量。
    :param x: 稠密张量
    :return: 稀疏张量的索引和值
    """
    indices = torch.nonzero(x).t()  # 获取非零元素的索引
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]  # 获取非零元素的值
    return indices, values


def cal_adj_mat_parameter(edge_per_node, x1, x2, metric="cosine"):
    """
    计算邻接矩阵的阈值参数。
    :param edge_per_node: 每个节点的期望边数
    :param data: 输入数据 (n, d)
    :param metric: 距离度量方法，目前仅支持 "cosine"
    :return: 阈值参数
    """
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(x1, x2)  # 计算余弦距离矩阵
    parameter = torch.sort(dist.reshape(-1)).values[edge_per_node * x1.shape[0]]  # 计算阈值
    return parameter.item()  # 转换为 Python 标量


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    """
    从距离矩阵生成邻接矩阵。
    :param dist: 距离矩阵 (n, n)
    :param parameter: 阈值参数
    :param self_dist: 是否排除自环
    :return: 邻接矩阵 (n, n)
    """
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()  # 根据阈值生成邻接矩阵
    if self_dist:
        g.fill_diagonal_(0)  # 排除自环
    return g


def gen_adj_mat_tensor(x1, x2, parameter, metric="cosine"):
    """
    生成邻接矩阵张量。
    :param data: 输入数据 (n, d)
    :param parameter: 阈值参数
    :param metric: 距离度量方法，目前仅支持 "cosine"
    :return: 邻接矩阵张量 (n, n)
    """
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(x1, x2)  # 计算余弦距离矩阵
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)  # 生成邻接矩阵
    adj = 1 - dist  # 将距离转换为相似度
    adj = adj * g  # 应用邻接矩阵掩码
    # adj = torch.max(adj, adj.t())  # 确保对称性
    adj = adj + torch.eye(adj.shape[0], device=adj.device)  # 添加自环
    return adj


def normalize_adj(adj):
    """
    对称归一化邻接矩阵。
    :param adj: 邻接矩阵 (n, n)
    :return: 归一化的邻接矩阵 (n, n)
    """
    D = torch.sum(adj, dim=1)  # 计算度矩阵
    D_inv_sqrt = torch.diag_embed(torch.pow(D, -0.5))  # 计算 D^(-1/2)
    adj_norm = torch.mm(D_inv_sqrt, torch.mm(adj, D_inv_sqrt))  # 对称归一化
    return adj_norm


def get_adj(x1, x2, k):
    """
    生成归一化的邻接矩阵。
    :param x: 输入数据 (n, d)
    :param k: 每个节点的期望边数
    :return: 归一化的邻接矩阵 (n, n)
    """
    p = cal_adj_mat_parameter(k, x1, x2)  # 计算阈值参数
    adj_gen = gen_adj_mat_tensor(x1, x2, p)  # 生成邻接矩阵
    adj_gen = normalize_adj(adj_gen)  # 归一化
    indices, values = to_sparse(adj_gen)
    return indices


def unsupervised_mask_by_patient_unified(x1, x2, x3, mask_rate=0.5):
    """
    无监督掩码：按患者编号进行掩码，确保所有模态掩码掉的患者一致
    :param x1: 模态1的特征 (batch_size, feature_dim)
    :param x2: 模态2的特征 (batch_size, feature_dim)
    :param x3: 模态3的特征 (batch_size, feature_dim)
    :param mask_rate: 掩码比例
    :return: 掩码后的特征 (x1_masked, x2_masked, x3_masked)
    """
    batch_size = x1.shape[0]
    mask = torch.ones(batch_size, 1)  # 初始化全1掩码 (batch_size, 1)

    # 随机选择一部分患者进行掩码
    num_mask = int(batch_size * mask_rate)  # 需要掩码的患者数量
    mask_indices = torch.randperm(batch_size)[:num_mask]  # 随机选择患者编号

    # 对选中的患者进行掩码
    mask[mask_indices] = 0  # 将这些患者的特征全部置为0

    # 对所有模态应用相同的掩码
    x1_masked = x1 * mask
    x2_masked = x2 * mask
    x3_masked = x3 * mask

    return x1_masked, x2_masked, x3_masked


def supervised_mask_by_label_unified(x1, x2, x3, y, mask_rate=0.3):
    y_pd = pd.DataFrame(y.cpu().numpy())
    label_loc = [group.index.to_numpy() for _, group in y_pd.groupby(0)]
    mask_nodes, keep_nodes = [], []
    for _ in label_loc:
        #
        num_nodes = len(_)
        perm = torch.randperm(num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        mask_nodes.append(_[perm[: num_mask_nodes]])
        keep_nodes.append(_[perm[num_mask_nodes:]])

    mask_nodes = np.concatenate(mask_nodes)
    keep_nodes = np.concatenate(keep_nodes)

    out_x1 = x1.clone()
    out_x1[0][mask_nodes] = 0.0

    out_x2 = x2.clone()
    out_x2[0][mask_nodes] = 0.0

    out_x3 = x3.clone()
    out_x3[0][mask_nodes] = 0.0

    return out_x1, out_x2, out_x3, (mask_nodes, keep_nodes)


# 在文件末尾添加
def drop_edges(edge_index, p=0.2):
    mask = torch.rand(edge_index.size(1)) > p
    return edge_index[:, mask]

def perturb_edge_attr(edge_attr, noise_level=0.1):
    return edge_attr + torch.randn_like(edge_attr) * noise_level

def subgraph_sampling(data, rate=0.8):
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.size(1)
        sample_size = int(num_edges * rate)
        perm = torch.randperm(num_edges)[:sample_size]
        data[edge_type].edge_index = data[edge_type].edge_index[:, perm]
    return data

