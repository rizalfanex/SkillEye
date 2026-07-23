import numpy as np

from imu_fusion import synthetic_imu_from_skeleton
from quality.keypoints import R_WRIST, R_ELBOW


def make_moving_clip(T=64):
    """Right wrist moves on a small circle around a fixed right elbow (left
    side stays at the origin, so dominant_wrist_index picks the right wrist,
    same convention as quality/test_phases.py)."""
    kpts = np.zeros((T, 17, 2), dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    kpts[:, R_ELBOW, 0] = 0.0
    kpts[:, R_ELBOW, 1] = 0.0
    kpts[:, R_WRIST, 0] = np.cos(t * 0.2)
    kpts[:, R_WRIST, 1] = np.sin(t * 0.2)
    return kpts


def test_output_shape():
    kpts = make_moving_clip(64)
    imu = synthetic_imu_from_skeleton(kpts)
    assert imu.shape == (64, 6)


def test_deterministic():
    kpts = make_moving_clip(64)
    imu1 = synthetic_imu_from_skeleton(kpts)
    imu2 = synthetic_imu_from_skeleton(kpts)
    np.testing.assert_array_equal(imu1, imu2)


def test_placeholder_channels_are_exactly_zero():
    kpts = make_moving_clip(64)
    imu = synthetic_imu_from_skeleton(kpts)
    assert np.all(imu[:, 2] == 0.0)  # accel_z
    assert np.all(imu[:, 3] == 0.0)  # gyro_x
    assert np.all(imu[:, 4] == 0.0)  # gyro_y


def test_derived_channels_are_non_constant():
    kpts = make_moving_clip(64)
    imu = synthetic_imu_from_skeleton(kpts)
    assert imu[:, 0].std() > 1e-6  # accel_x
    assert imu[:, 1].std() > 1e-6  # accel_y
    assert imu[:, 5].std() > 1e-6  # gyro_z


def test_static_clip_gives_all_zero_signal():
    kpts = np.zeros((64, 17, 2), dtype=np.float32)
    imu = synthetic_imu_from_skeleton(kpts)
    assert np.all(imu == 0.0)
