import numpy as np
import torch

from stroke_dataset import FIXED_T, resample_time
from imu_fusion import synthetic_imu_from_skeleton
from train_beginner_expert_fusion_prototype import FusionSkillDataset


def make_fake_record(skill_level, subject_id, T=30, seed=0):
    """Small in-memory fake record -- no real skeleton files needed. The motion
    doesn't need to be realistic, just non-degenerate (random, not all-zero) so
    the derived IMU signal isn't trivially all-zero too."""
    rng = np.random.RandomState(seed)
    kpts = rng.uniform(-1, 1, size=(T, 17, 2)).astype(np.float32)
    return {"kpts": kpts, "skill_level": skill_level, "subject_id": subject_id}


def test_fusion_dataset_item_shapes():
    records = [make_fake_record("expert", subject_id=7, seed=0)]
    ds = FusionSkillDataset(records, augment=False)

    skeleton_tensor, imu_tensor, label, subject_id = ds[0]

    assert skeleton_tensor.shape == (4, FIXED_T, 17)
    assert imu_tensor.shape == (6, FIXED_T)
    assert label == 1
    assert subject_id == 7


def test_fusion_dataset_beginner_label():
    records = [make_fake_record("beginner", subject_id=3, seed=1)]
    ds = FusionSkillDataset(records, augment=False)

    _, _, label, subject_id = ds[0]

    assert label == 0
    assert subject_id == 3


def test_fusion_dataset_imu_matches_direct_call_on_same_resampled_clip():
    """Regression check: the skeleton and IMU tensors returned by __getitem__
    must be derived from the SAME (resampled) clip view, not two independently
    randomized ones. With augment=False, the dataset's internal path is just
    resample_time(kpts, FIXED_T) -> synthetic_imu_from_skeleton(...); replicate
    that directly and compare against what the dataset produced (accounting for
    the dataset's (T, 6) -> (6, T) permute)."""
    records = [make_fake_record("expert", subject_id=42, seed=2)]
    ds = FusionSkillDataset(records, augment=False)

    _, imu_tensor, _, _ = ds[0]

    expected_kpts = resample_time(records[0]["kpts"], FIXED_T)
    expected_imu = synthetic_imu_from_skeleton(expected_kpts)  # (T, 6)
    expected_imu_tensor = torch.from_numpy(
        expected_imu.astype(np.float32)).permute(1, 0).contiguous()  # (6, T)

    torch.testing.assert_close(imu_tensor, expected_imu_tensor)
