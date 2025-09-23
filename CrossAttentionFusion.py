import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_dim, num_heads=2, use_contrastive=False):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.use_contrastive = use_contrastive

        # 更合理的注意力层设计
        self.attention1 = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
        self.attention3 = nn.MultiheadAttention(embed_dim=in_dim*3, num_heads=num_heads)

        # 添加层归一化和残差连接
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim*3)

        # 更合理的融合投影层
        self.proj = nn.Sequential(
            nn.Linear(in_dim * 5, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim)
        )

    def forward(self, x1, x2, x3, y=None, testing=False):
        # 确保输入维度正确 [seq_len, batch_size, in_dim]
        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            x3 = x3.unsqueeze(1)

        # 注意力1: x1作为query，x2作为key/value
        attn1, _ = self.attention1(x1, x2, x2)
        # attn1 = self.norm1(x1 + attn1)  # 残差连接

        # 注意力2: x2作为query，x3作为key/value
        attn2, _ = self.attention2(x1, x3, x3)
        # attn2 = self.norm2(x2 + attn2)

        # 注意力3: x3作为query，x1作为key/value
        x_ = torch.concatenate([x1, x2, x3], dim=-1)
        attn3, _ = self.attention3(x_, x_, x_)
        # attn3 = self.norm3(x_ + attn3)

        # 特征融合
        fused = torch.cat([attn1, attn2, attn3], dim=-1)
        fused = self.proj(fused)

        # 恢复原始维度
        if x1.size(1) == 1:
            fused = fused.squeeze(1)

        return fused