# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class ScanCNNEncoder(nn.Module):
    def __init__(self, d=64, keep_d=3, activation="elu"):
        """
        CNN模块用于处理网格点云的z值
        Args:
            d: MHA模块的K-V维度，默认为64
        """
        super(ScanCNNEncoder, self).__init__()
        # 计算输出通道数
        output_channels = d - keep_d  # 61 when d=64
        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels=1,        # 输入通道：z值
            out_channels=16,      # 输出通道：16
            kernel_size=5,        # 卷积核大小：5×5
            padding=2,            # 填充：2，保持11×17的维度不变
            stride=1
        )
        # 第二层卷积
        self.conv2 = nn.Conv2d(
            in_channels=16,       # 输入通道：16
            out_channels=output_channels,  # 输出通道：d-3
            kernel_size=5,        # 卷积核大小：5×5
            padding=2,            # 填充：2，保持11×17的维度不变
            stride=1
        )
        # 激活函数（根据常见实践添加，原描述未明确指定）
        self.activation_layer = resolve_nn_activation(activation)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 [batch_size, 1, 11, 17] 或 [batch_size, 11, 17, 1]
        Returns:
            输出张量，形状为 [batch_size, d-3, 11, 17]
        """
        # 如果输入是 [batch_size, 11, 17, 1] 格式，调整为 [batch_size, 1, 11, 17]
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        # 第一层卷积 + 激活
        x = self.activation_layer(self.conv1(x))
        # 第二层卷积
        x = self.conv2(x)
        return x

    def init_weights(self, scales: float | tuple[float]):
        """
        初始化卷积层权重
        Args:
            scales: 权重初始化的缩放因子，可以是单值或元组
        """
        def get_scale(idx) -> float:
            # 如果scales是元组或列表，则按索引获取对应层的缩放因子
            # 否则返回相同的缩放因子
            if isinstance(scales, (list, tuple)):
                # 确保索引不超出范围
                return scales[idx] if idx < len(scales) else scales[-1]
            return scales
        
        # 收集所有卷积层
        conv_layers = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
        # 对每个卷积层进行初始化
        for idx, conv_layer in enumerate(conv_layers):
            # 使用正交初始化方法设置权重
            nn.init.orthogonal_(conv_layer.weight, gain=get_scale(idx))
            # 将偏置初始化为零
            if conv_layer.bias is not None:
                nn.init.zeros_(conv_layer.bias)
