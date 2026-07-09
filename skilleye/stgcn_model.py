"""
Compact ST-GCN (Yan et al. 2018 style) for THETIS skeleton clips.

Input:  (N, C=2, T, V=17) hip-anchored/torso-scaled COCO-17 keypoint sequences
Output: (N, num_classes) logits

Kept deliberately small: THETIS gives ~1900 usable clips after cleaning, not
the tens of thousands ST-GCN was originally tuned on, so a few spatio-temporal
blocks with modest channel widths is the right capacity, not a deeper net.
"""

import torch
import torch.nn as nn

# COCO-17 order (see skeleton_pipeline.py):
# 0 nose,1 l_eye,2 r_eye,3 l_ear,4 r_ear,5 l_sho,6 r_sho,7 l_elb,8 r_elb,
# 9 l_wri,10 r_wri,11 l_hip,12 r_hip,13 l_knee,14 r_knee,15 l_ank,16 r_ank
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]
NUM_JOINTS = 17


def build_adjacency():
    """Symmetric-normalized adjacency (A_hat = D^-1/2 (A+I) D^-1/2), COCO-17 skeleton graph."""
    A = torch.eye(NUM_JOINTS)
    for i, j in COCO17_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class SpatialGraphConv(nn.Module):
    """1x1 conv (channel mixing) followed by graph aggregation over the fixed adjacency."""

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer("A", A)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        x = self.conv(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, temporal_kernel=9, stride=1, residual=True):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        if self.residual is not None:
            x = x + res
        return self.relu(x)


class STGCN(nn.Module):
    def __init__(self, num_classes, in_channels=2, base_channels=32):
        super().__init__()
        A = build_adjacency()
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        c = base_channels
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, c, A, residual=False),
            STGCNBlock(c, c, A),
            STGCNBlock(c, c * 2, A, stride=2),
            STGCNBlock(c * 2, c * 2, A),
            STGCNBlock(c * 2, c * 4, A, stride=2),
            STGCNBlock(c * 4, c * 4, A),
        ])
        self.fc = nn.Linear(c * 4, num_classes)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=[2, 3])  # global average pool over T, V
        return self.fc(x)
