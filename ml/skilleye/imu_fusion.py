"""
Late-fusion prototype: adds a racket-mounted-IMU-shaped input branch to the
beginner/expert STGCN classifier. No MPU6050 hardware exists yet (see
docs/superpowers/specs/2026-07-23-imu-fusion-prototype-design.md), so
synthetic_imu_from_skeleton() below is a clearly-labeled placeholder signal,
not a real sensor reading -- its purpose is to prove the fusion architecture
and code path work, not to claim any accuracy result. When real MPU6050 logs
exist, replacing this function's call site with a real-data loader (resampled
to the same T the same way) is the only change needed.
"""
import numpy as np
import torch
import torch.nn as nn

from quality.phases import dominant_wrist_index
from quality.keypoints import R_WRIST, L_ELBOW, R_ELBOW
from stgcn_model import STGCN


def synthetic_imu_from_skeleton(kpts):
    """kpts: (T, 17, 2) skeleton, already resampled to the model's fixed T.
    Returns (T, 6): [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z], the
    standard MPU6050 channel order (chosen so a real sensor log can later
    drop in with no remapping).

    Only channels honestly derivable from 2D pose carry a signal:
    accel_x/y (indices 0, 1) from the dominant wrist's 2nd finite difference
    (acceleration), gyro_z (index 5) from the forearm segment's angular
    velocity. accel_z, gyro_x, gyro_y (indices 2, 3, 4) are exact zeros --
    2D pose has no depth/out-of-plane rotation to derive them from honestly.
    This signal is, by construction, redundant with the skeleton branch's own
    position/velocity channels -- it is a code-path placeholder, not an
    independent modality.
    """
    T = kpts.shape[0]
    wrist_idx = dominant_wrist_index(kpts)
    elbow_idx = R_ELBOW if wrist_idx == R_WRIST else L_ELBOW

    wrist_pos = kpts[:, wrist_idx]  # (T, 2)
    velocity = np.zeros_like(wrist_pos)
    velocity[1:] = wrist_pos[1:] - wrist_pos[:-1]
    accel = np.zeros_like(wrist_pos)
    accel[1:] = velocity[1:] - velocity[:-1]

    elbow_pos = kpts[:, elbow_idx]
    forearm_vec = wrist_pos - elbow_pos
    angle = np.arctan2(forearm_vec[:, 1], forearm_vec[:, 0])
    gyro_z = np.zeros((T,), dtype=np.float32)
    raw_diff = angle[1:] - angle[:-1]
    gyro_z[1:] = np.arctan2(np.sin(raw_diff), np.cos(raw_diff))

    imu = np.zeros((T, 6), dtype=np.float32)
    imu[:, 0] = accel[:, 0]
    imu[:, 1] = accel[:, 1]
    imu[:, 5] = gyro_z
    return imu


class IMUEncoder(nn.Module):
    """Small 1D-CNN over the 6-channel IMU stream, pooled to one feature
    vector per clip. Kept small deliberately, same reasoning as STGCN's own
    size: this prototype has no real IMU data volume to justify more
    capacity."""

    def __init__(self, in_channels=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (N, in_channels, T) -> (N, hidden)
        x = self.net(x)
        return x.mean(dim=2)


class FusedBeginnerExpertModel(nn.Module):
    """Late-fusion prototype: the existing STGCN skeleton branch (via
    extract_features) plus an IMUEncoder branch, concatenated before one
    classification head. See module docstring for the synthetic-data
    caveat -- this architecture is real, its current training data is not."""

    def __init__(self, num_classes=2, skeleton_channels=4, imu_channels=6,
                 skeleton_base_channels=32, imu_hidden=32):
        super().__init__()
        self.skeleton_branch = STGCN(
            num_classes=num_classes, in_channels=skeleton_channels,
            base_channels=skeleton_base_channels,
        )
        self.imu_branch = IMUEncoder(in_channels=imu_channels, hidden=imu_hidden)
        self.fc = nn.Linear(skeleton_base_channels * 4 + imu_hidden, num_classes)

    def forward(self, x_skeleton, x_imu):
        skeleton_feats = self.skeleton_branch.extract_features(x_skeleton)
        imu_feats = self.imu_branch(x_imu)
        combined = torch.cat([skeleton_feats, imu_feats], dim=1)
        return self.fc(combined)
