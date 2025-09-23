import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import HeteroData
from base_layer import HeteroGNN
import utlis
from torch_geometric.utils import degree
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from CrossAttentionFusion import CrossAttentionFusion


class HeteroGraphAugmentor:
    def __init__(self, p_tau=0.7):
        self.p_tau = p_tau
        # 预分配一些内存
        self._temp_masks = {}

    def compute_node_drop_prob(self, x, edge_index):
        """向量化计算节点概率"""
        with torch.no_grad():
            deg = degree(edge_index[0], num_nodes=x.size(0), dtype=torch.float)
            deg_max, deg_mean = deg.max(), deg.mean()
            denominator = deg_max - deg_mean + 1e-6
            return torch.clamp((deg_max - deg) / denominator, max=self.p_tau)

    def structure_augmentation(self, edge_index, node_p):
        """优化后的结构增强"""
        src, dst = edge_index
        device = edge_index.device

        # 向量化计算边概率
        edge_p = (node_p[src] + node_p[dst]) / 2
        mask = torch.rand(edge_p.size(0), device=device) > edge_p

        # 快速处理孤立节点
        unique_nodes, counts = torch.unique(torch.cat([src, dst]), return_counts=True)
        isolated_nodes = unique_nodes[counts == 1]

        if len(isolated_nodes) > 0:
            # 为孤立节点保留至少一条边
            isolated_mask = torch.isin(src, isolated_nodes) | torch.isin(dst, isolated_nodes)
            mask[isolated_mask] = True  # 强制保留孤立节点的边

        return edge_index[:, mask]

    def augment_hetero_data(self, data):
        """并行化异质图增强"""
        augmented_data = HeteroData()

        # 并行处理节点特征
        for node_type in data.node_types:
            x = data[node_type].x
            edge_index = data[node_type, 'in', node_type].edge_index
            node_p = self.compute_node_drop_prob(x, edge_index)

            # 使用原地操作减少内存分配
            mask = torch.bernoulli(1 - node_p.unsqueeze(1), out=torch.empty_like(x))
            augmented_data[node_type].x = x * mask

        # 并行处理边类型
        edge_types = list(data.edge_types)
        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index

            # 预计算节点概率
            src_p = self.compute_node_drop_prob(data[src_type].x,
                                                data[src_type, 'in', src_type].edge_index)
            dst_p = self.compute_node_drop_prob(data[dst_type].x,
                                                data[dst_type, 'in', dst_type].edge_index)

            # 向量化边处理
            avg_p = (src_p[edge_index[0]] + dst_p[edge_index[1]]) / 2
            mask = torch.rand(avg_p.size(0), device=edge_index.device) > avg_p
            augmented_data[edge_type].edge_index = edge_index[:, mask]

        return augmented_data


class MultiViewHeteroGNNContrastiveLearning(nn.Module):
    def __init__(self, in_dim, k2, k3, dropout=0.5, tau=0.5, p_tau=0.6):
        super().__init__()
        self.k2 = k2
        self.k3 = k3
        self.MultiViewHeteroGNNFusionLearning = HeteroGNN(in_dim, dropout)
        self.augmentor = HeteroGraphAugmentor(p_tau=p_tau)
        self.tau = nn.Parameter(torch.tensor(tau))
        # fusion
        self.cross_attention = CrossAttentionFusion(in_dim[-1])

    def forward(self, x1, x2, x3, y=None, testing=False):
        # 构图
        data = self.create_data(x1, x2, x3)
        # 特征提取
        x1, x2, x3 = self.MultiViewHeteroGNNFusionLearning(data)
        fusion = self.cross_attention(x1, x2, x3)
        loss = 0
        # 对比损失计算
        if not testing:
            # 特征增强
            aug_data = self.augmentor.augment_hetero_data(data)
            x1_aug, x2_aug, x3_aug = self.MultiViewHeteroGNNFusionLearning(aug_data)
            fusion_ = self.cross_attention(x1_aug, x2_aug, x3_aug)
            # 无监督对比学习
            loss += self.contrastive_loss(torch.cat([x1, x2, x3]), torch.cat([x1_aug, x2_aug, x3_aug]))
            # 有监督对比学习
            all_fusions = torch.cat([fusion, fusion_], dim=0)  # [2N, D]
            all_labels = torch.cat([y, y], dim=0)  # [2N]（因为 fusion 和 fusion_ 来自同一样本，标签相同）
            loss += self.supervised_contrastive_loss(all_fusions, all_labels)

        return fusion, loss

    def create_data(self, x1, x2, x3):
        data = HeteroData()
        data['x1'].x = x1
        data['x2'].x = x2
        data['x3'].x = x3
        # 模态1-模态1
        data['x1', 'in', 'x1'].edge_index = utlis.get_adj(x1, x1, self.k2)
        # 模态2-模态2
        data['x2', 'in', 'x2'].edge_index = utlis.get_adj(x2, x2, self.k2)
        # 模态3-模态3
        data['x3', 'in', 'x3'].edge_index = utlis.get_adj(x3, x3, self.k2)
        # 模态1-模态2
        data['x1', 'between', 'x2'].edge_index = utlis.get_adj(x1, x2, self.k3)
        # 模态1-模态3
        data['x1', 'between', 'x3'].edge_index = utlis.get_adj(x1, x3, self.k3)
        # 模态2-模态3
        data['x2', 'between', 'x3'].edge_index = utlis.get_adj(x2, x3, self.k3)
        return data

    def contrastive_loss(self, z_orig, z_aug):
        # z_orig: 原始特征 [N, D]
        # z_aug: 增强特征 [N, D]
        z_orig = F.normalize(z_orig, dim=1)
        z_aug = F.normalize(z_aug, dim=1)

        # 模态内对比（原始-增强）
        intra_sim = torch.exp(torch.mm(z_orig, z_aug.t()) / self.tau)
        pos_mask = torch.eye(z_orig.size(0), device=z_orig.device)
        intra_loss = -torch.log(intra_sim[pos_mask == 1] / intra_sim.sum(1)).mean()

        # 跨模态对比（可选）
        cross_sim = torch.exp(torch.mm(z_orig, z_orig.t()) / self.tau)
        cross_loss = -torch.log(cross_sim[pos_mask == 1] / (cross_sim.sum(1) - cross_sim.diag())).mean()

        return 1 * intra_loss + 0 * cross_loss

    def supervised_contrastive_loss(self, features, labels, temperature=0.1):
        """
        features: [2N, D]（包含 fusion 和 fusion_）
        labels: [2N]（对应的标签）
        temperature: 温度参数，控制对比学习的难度
        """
        features = F.normalize(features, dim=1)  # L2 归一化
        similarity_matrix = torch.matmul(features, features.T) / temperature  # [2N, 2N]

        # 构建 mask: 相同类别的样本为正样本对
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))  # [2N, 2N]

        # 排除自身对比（可选）
        self_mask = torch.eye(labels.size(0), dtype=torch.bool, device=features.device)
        mask = mask & (~self_mask)  # 去掉自身对比

        # 计算正样本对的相似度
        positives = similarity_matrix[mask]

        # 计算负样本对的相似度
        negatives = similarity_matrix[~mask]

        # 计算损失（类似 InfoNCE loss）
        numerator = torch.exp(positives).sum()
        denominator = torch.exp(similarity_matrix).sum(dim=1) - torch.exp(similarity_matrix.diag())
        loss = -torch.log(numerator / denominator).mean()

        return loss