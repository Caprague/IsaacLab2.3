import torch
import torch.nn as nn


class DepthImageEncoder(nn.Module):
    def __init__(self, activation: str = "relu"):
        super().__init__()

        act_fn = nn.ReLU if activation == "relu" else nn.ELU

        self.horizontal_conv = nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 0))

        self.vertical_conv = nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0))

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            act_fn(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            act_fn(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)

        x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)

        x = self.horizontal_conv(x)
        x = self.vertical_conv(x)

        x = self.fusion_conv(x)

        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        min_feat = -self.max_pool(-x).flatten(1)

        return torch.cat([avg_feat, max_feat, min_feat], dim=-1)