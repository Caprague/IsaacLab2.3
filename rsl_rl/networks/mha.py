# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(self, d=64, num_heads=16, dropout=0.1):
        """
        多头自注意力模块
        Args:
            d: MHA模块的K-V维度，默认为64
            num_heads: 头数，默认为4
        """
        super(MHA, self).__init__()
        self.embed_dim = d
        self.num_heads = num_heads
        self.dropout = dropout

        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
    
    def forward(self, X, Y):
        """
        前向传播
        Args:
            X: 输入张量，形状为 [batch_size, q_len, d]
            Y: 输入张量，形状为 [batch_size, kv_len, d]
        Returns:
            输出张量，形状为 [batch_size, q_len, d]
            注意力权重，形状为 [batch_size, num_heads, q_len, kv_len] (average_attn_weights=False)
                        OR  [batch_size, q_len, kv_len] (average_attn_weights=True, Default)
        """
        output, attn_weights = self.mha(X, Y, Y, need_weights=True, average_attn_weights=False)
        return output, attn_weights
