import torch
from base_layer import GateSelect, EncoderGAT
import utlis


class DynamicalGraphLearning(torch.nn.Module):
    def __init__(self, feature_dim, in_dim, class_num, k, dropout):
        super(DynamicalGraphLearning, self).__init__()
        self.k = k  # 图平均保留多少个邻居节点
        self.gate = GateSelect(feature_dim, dropout)  # 特征筛选模块
        self.encoder = EncoderGAT(feature_dim, in_dim, dropout)  # 编码器
        self.cl = torch.nn.Linear(in_dim[-1], class_num)  # 辅助分类器

    def forward(self, x, y, testing=False):
        # x = torch.nn.functional.normalize(x, dim=1)
        att_score, x = self.gate(x)  # 特征筛选
        edge_index = utlis.get_adj(x, x, self.k)
        representation = self.encoder(x, edge_index)
        pre = self.cl(representation)
        loss = 0
        if not testing:
            criterion = torch.nn.CrossEntropyLoss()
            loss_pre = criterion(pre, y)  # 分类损失
            # loss_fc = torch.mean(att_score)  # 特征筛选损失
            loss_fc = torch.mean(torch.norm(att_score, p=1))  # L1正则促进稀疏

            loss = 1.0 * loss_pre + 0.000001 * loss_fc  # 加权损失
        return representation, loss
