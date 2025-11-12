# -----------------------------------------------------------
# core/encoder.py
# this encoder turns math formula images into feature sequences.
# uses coordconv (adds x/y coords), depthwise convs, SE-blocks,
# residual connections and droppath for regularization.
# mostly downsamples height but keeps width to preserve reading order.
# final output: (B, S, D) for the transformer decoder.
# -----------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple



class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        return x + (torch.rand(shape, device=x.device) < self.p).float() * (-x / keep)

def make_coord(batch: int, h: int, w: int, device) -> torch.Tensor:

    y = torch.linspace(-1, 1, steps=h, device=device).view(1, 1, h, 1).expand(batch, 1, h, w)
    x = torch.linspace(-1, 1, steps=w, device=device).view(1, 1, 1, w).expand(batch, 1, h, w)
    return torch.cat([y, x], dim=1)


class ConvBNAct(nn.Module):
    def __init__(self, ci, co, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(co)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class SE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class DSConv(nn.Module):
    def __init__(self, ci, co, s=1):
        super().__init__()
        self.dw = ConvBNAct(ci, ci, k=3, s=s, p=1, groups=ci)
        self.pw = ConvBNAct(ci, co, k=1, s=1, p=0, groups=1)
    def forward(self, x): return self.pw(self.dw(x))

class ResidualDS(nn.Module):
    def __init__(self, c, s=1, droppath=0.0, use_se=True):
        super().__init__()
        self.conv1 = DSConv(c, c, s=s)
        self.conv2 = DSConv(c, c, s=1)
        self.se = SE(c) if use_se else nn.Identity()
        self.short = nn.Identity()
        self.dp = DropPath(droppath)
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        y = self.se(y)
        return x + self.dp(y)

class DownStage(nn.Module):

    def __init__(self, ci, co, stride_hw: Tuple[int,int]=(2,2)):
        super().__init__()
        sh, sw = stride_hw
        self.down = ConvBNAct(ci, co, k=3, s=1, p=1)

        self.pool = nn.AvgPool2d(kernel_size=(sh, sw), stride=(sh, sw))
    def forward(self, x):
        x = self.down(x)
        return self.pool(x)


class CNNEncoder(nn.Module):
    def __init__(self, d_model: int = 384, base_c: int = 64, depth: Tuple[int,int,int]=(2,2,2),
                 droppath_max: float = 0.1, coordconv: bool = True):

        super().__init__()
        self.coordconv = coordconv
        stem_in = 1 + (2 if coordconv else 0)
        self.stem = nn.Sequential(
            ConvBNAct(stem_in, base_c, k=3, s=1, p=1),
            ConvBNAct(base_c, base_c, k=3, s=1, p=1),
        )

        C1 = base_c
        C2 = base_c * 2
        C3 = base_c * 4


        self.stage1_down = DownStage(C1, C2, stride_hw=(2,2))
        self.stage2_down = DownStage(C2, C3, stride_hw=(2,1))


        total_blocks = sum(depth)
        idx = 0
        self.stage1 = nn.Sequential(*[
            ResidualDS(C2, s=1, droppath=droppath_max * (idx:=idx+1)/total_blocks) for _ in range(depth[0])
        ])
        self.stage2 = nn.Sequential(*[
            ResidualDS(C3, s=1, droppath=droppath_max * (idx:=idx+1)/total_blocks) for _ in range(depth[1])
        ])
        self.stage3 = nn.Sequential(*[
            ResidualDS(C3, s=1, droppath=droppath_max * (idx:=idx+1)/total_blocks) for _ in range(depth[2])
        ])

        self.proj = nn.Linear(C3, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, _, H, W = x.shape
        if self.coordconv:
            coord = make_coord(B, H, W, x.device)
            x = torch.cat([x, coord], dim=1)

        x = self.stem(x)
        x = self.stage1_down(x)
        x = self.stage1(x)

        x = self.stage2_down(x)
        x = self.stage2(x)

        x = self.stage3(x)


        x = x.mean(dim=2)
        x = x.permute(0, 2, 1).contiguous()
        return self.proj(x)