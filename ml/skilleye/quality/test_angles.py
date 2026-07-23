import numpy as np

from quality.angles import (
    joint_angle_series, trunk_rotation_series, compute_all_angles,
    phase_mean_angles, JOINT_DEFINITIONS,
)
from quality.keypoints import L_SHOULDER, R_SHOULDER, L_ELBOW, L_WRIST, L_HIP, R_HIP


def test_joint_angle_series_right_angle():
    # shoulder at (0,1), elbow at (0,0), wrist at (1,0) -> 90 degrees at the elbow
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 1)
    kpts[0, L_ELBOW] = (0, 0)
    kpts[0, L_WRIST] = (1, 0)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert result.shape == (1,)
    assert np.isclose(result[0], np.pi / 2, atol=1e-5)


def test_joint_angle_series_straight_arm_is_pi():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 1)
    kpts[0, L_ELBOW] = (0, 0)
    kpts[0, L_WRIST] = (0, -1)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert np.isclose(result[0], np.pi, atol=1e-5)


def test_joint_angle_series_empty_input():
    kpts = np.zeros((0, 17, 2), dtype=np.float32)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert result.shape == (0,)


def test_trunk_rotation_series_parallel_lines_is_zero():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 0)
    kpts[0, R_SHOULDER] = (1, 0)
    kpts[0, L_HIP] = (0, -1)
    kpts[0, R_HIP] = (1, -1)
    result = trunk_rotation_series(kpts)
    assert np.isclose(result[0], 0.0, atol=1e-5)


def test_compute_all_angles_has_expected_keys_and_shapes():
    kpts = np.zeros((3, 17, 2), dtype=np.float32)
    result = compute_all_angles(kpts)
    assert set(result.keys()) == set(JOINT_DEFINITIONS.keys()) | {"trunk_rotation"}
    for series in result.values():
        assert series.shape == (3,)


def test_phase_mean_angles_returns_none_for_empty_phase():
    kpts = np.zeros((0, 17, 2), dtype=np.float32)
    result = phase_mean_angles(kpts)
    assert all(v is None for v in result.values())


def test_phase_mean_angles_returns_float_for_nonempty_phase():
    kpts = np.zeros((3, 17, 2), dtype=np.float32)
    kpts[:, L_SHOULDER] = (0, 1)
    kpts[:, L_ELBOW] = (0, 0)
    kpts[:, L_WRIST] = (1, 0)
    result = phase_mean_angles(kpts)
    assert isinstance(result["left_elbow"], float)
    assert np.isclose(result["left_elbow"], np.pi / 2, atol=1e-5)
