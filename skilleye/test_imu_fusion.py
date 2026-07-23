import numpy as np
import torch

from imu_fusion import synthetic_imu_from_skeleton, IMUEncoder, FusedBeginnerExpertModel
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


def test_gyro_z_angle_wrap_no_spurious_jump():
    """Regression test for branch-cut crossing in gyro_z computation.
    Forearm angle crosses from just under +pi to just under -pi
    (small true rotation, but would spike to ~2*pi without wrapping fix).
    """
    kpts = np.zeros((2, 17, 2), dtype=np.float32)
    # Elbow at origin
    kpts[:, R_ELBOW, 0] = 0.0
    kpts[:, R_ELBOW, 1] = 0.0
    # Wrist: frame 0 at angle just under +pi, frame 1 at angle just under -pi
    # This is a small rotation of ~0.2 rad across the wrap
    angle_0 = np.pi - 0.1
    angle_1 = -np.pi + 0.1
    kpts[0, R_WRIST, 0] = np.cos(angle_0)
    kpts[0, R_WRIST, 1] = np.sin(angle_0)
    kpts[1, R_WRIST, 0] = np.cos(angle_1)
    kpts[1, R_WRIST, 1] = np.sin(angle_1)

    imu = synthetic_imu_from_skeleton(kpts)
    # gyro_z[1] should be small (the true rotation ~0.2 rad), not ~2*pi
    assert np.abs(imu[1, 5]) < 0.5, f"gyro_z[1] = {imu[1, 5]}, expected small magnitude"


def test_imu_encoder_output_shape():
    encoder = IMUEncoder()
    x = torch.randn(4, 6, 64)
    out = encoder(x)
    assert out.shape == (4, 32)
    assert not torch.isnan(out).any()


def test_fused_model_output_shape():
    model = FusedBeginnerExpertModel(num_classes=2)
    x_skeleton = torch.randn(4, 4, 64, 17)
    x_imu = torch.randn(4, 6, 64)
    logits = model(x_skeleton, x_imu)
    assert logits.shape == (4, 2)


def test_fused_model_gradient_reaches_imu_branch():
    model = FusedBeginnerExpertModel(num_classes=2)
    x_skeleton = torch.randn(4, 4, 64, 17)
    x_imu = torch.randn(4, 6, 64)
    labels = torch.tensor([0, 1, 0, 1])

    logits = model(x_skeleton, x_imu)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    imu_grad_norms = [
        p.grad.norm().item() for p in model.imu_branch.parameters() if p.grad is not None
    ]
    assert len(imu_grad_norms) > 0
    assert all(g == g for g in imu_grad_norms)  # no NaNs (NaN != NaN)
    assert sum(imu_grad_norms) > 0.0
