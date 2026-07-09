import numpy as np

from quality.phases import dominant_wrist_index, detect_contact_frame, split_phases, PHASES
from quality.keypoints import R_WRIST


def make_kpts(T, wrist_positions):
    """Builds a minimal (T, 17, 2) array with all joints static at the origin
    except the right wrist, which moves through wrist_positions (list of
    (x, y) tuples, length T)."""
    kpts = np.zeros((T, 17, 2), dtype=np.float32)
    for t in range(T):
        kpts[t, R_WRIST] = wrist_positions[t]
    return kpts


def test_dominant_wrist_index_picks_the_moving_wrist():
    T = 10
    positions = [(float(t), 0.0) for t in range(T)]  # right wrist moves, left stays at 0
    kpts = make_kpts(T, positions)
    assert dominant_wrist_index(kpts) == R_WRIST


def test_detect_contact_frame_finds_speed_peak():
    # right wrist accelerates steadily, has one big jump at t=5, then slows down
    T = 11
    positions = []
    x = 0.0
    for t in range(T):
        step = 1.0 if t < 5 else (5.0 if t == 5 else 0.2)
        x += step
        positions.append((x, 0.0))
    kpts = make_kpts(T, positions)
    contact = detect_contact_frame(kpts)
    assert 3 <= contact <= 7  # near the big jump between frame 4 and frame 5


def test_detect_contact_frame_falls_back_on_short_clips():
    kpts = np.zeros((2, 17, 2), dtype=np.float32)
    assert detect_contact_frame(kpts) == 1  # T // 2


def test_split_phases_returns_all_three_phases_covering_the_full_clip():
    T = 20
    positions = [(float(t), 0.0) for t in range(T)]
    kpts = make_kpts(T, positions)
    result = split_phases(kpts)
    assert set(result.keys()) == set(PHASES)
    total_frames = sum(p.shape[0] for p in result.values())
    assert total_frames == T
