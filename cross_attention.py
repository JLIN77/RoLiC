import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q_feat, kv_feat):

        B, C, H, W = q_feat.shape
        N = H * W

        Q = self.q_proj(q_feat)
        K = self.k_proj(kv_feat)
        V = self.v_proj(kv_feat)

        Q = Q.flatten(2).transpose(1, 2)     # B,N,C
        K = K.flatten(2).transpose(1, 2)     # B,N,C
        V = V.flatten(2).transpose(1, 2)     # B,N,C

        attn = torch.matmul(
            Q,
            K.transpose(-1, -2)
        ) / math.sqrt(C)

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(
            B, C, H, W
        )

        return out, attn