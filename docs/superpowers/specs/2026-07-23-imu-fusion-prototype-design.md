# IMU (MPU6050) Sensor-Fusion Prototype — Design

Date: 2026-07-23
Status: Approved for implementation

## Problem

In a team discussion (recorded, transcribed via Whisper), the user's teammate proposed
instrumenting the racket with an MPU6050 (accelerometer + gyroscope, 6-axis) to help the
system distinguish beginner from expert technique — motivated by a real, already-documented
limitation of this project: THETIS is a single frontal 2D camera at ~17-19fps
([[thetis-dataset-limitations]]), which cannot see racket-face angle or wrist-snap dynamics
at contact. Those are exactly the kind of high-frequency, off-camera-plane signals a
wrist/racket-mounted IMU captures directly.

No hardware has been ordered yet. The deadline (2026-09-01) is roughly 5.5 weeks out, and
this project's existing headline numbers (stroke classifier 81.7% ± 4.9%, beginner/expert
82.4% ± 3.8%, both 5-fold subject-disjoint cross-validated) are the result of real,
hard-won validation work that must not be diluted or put at risk by a new, unvalidated
feature.

## Approach: late-fusion prototype on synthetic IMU data, kept fully separate from the validated models

Build a second, standalone model — a small IMU-encoding branch fused with the existing
beginner/expert `STGCN` — that proves the *architecture and code path* work end to end.
Because no real sensor data exists yet, the IMU input is a clearly-labeled synthetic proxy
derived from the skeleton itself. This means the prototype **cannot and does not claim any
accuracy improvement**: the synthetic signal is, by construction, redundant with
information already inside the skeleton branch, so any accuracy delta would be circular,
not evidence of anything. This mirrors the project's established v1/v2/v3 pattern of
never overstating a result and keeping prior validated iterations alongside new ones
rather than overwriting them.

The prototype's actual deliverable is: (1) a fusion architecture ready to accept real
sensor data through the exact same interface once hardware exists, and (2) a documented
data-collection protocol so that swap-in requires no architecture rework.

## Components

### 1. `STGCN` feature extraction refactor (`stgcn_model.py`)

Split `STGCN.forward()` into:
- `extract_features(x)`: everything up to (not including) the final `fc` — returns the
  pooled `(N, base_channels*4)` = `(N, 128)` vector.
- `forward(x)`: unchanged externally — calls `extract_features` then `self.fc`.

Purely a refactor. Every existing caller (`train_stroke_classifier.py`,
`train_beginner_expert_stgcn.py`, `cross_validate.py`, `app.py`) keeps working unmodified,
and the validated cross-validation numbers are untouched.

### 2. Synthetic IMU signal (`imu_fusion.py`: `synthetic_imu_from_skeleton`)

Input: normalized skeleton clip `(T, 17, 2)`. Output: `(T, 6)`, channel order
`[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]` — the standard MPU6050 output
order, chosen specifically so a real sensor log can be substituted with no remapping:

- `accel_x, accel_y` (indices 0, 1): second finite difference (acceleration) of the
  dominant wrist keypoint position (`quality.phases.dominant_wrist_index`), arbitrary
  units (skeleton is normalized, not metric) — genuinely derived, but redundant with the
  skeleton branch's own position/velocity channels.
- `gyro_z` (index 5): time-derivative of the forearm segment's 2D orientation angle
  (`atan2(wrist - elbow)`) — genuinely derived in-plane angular velocity.
- `accel_z, gyro_x, gyro_y` (indices 2, 3, 4): **exact zeros**. 2D pose has no
  depth/out-of-plane rotation information to derive these from honestly. This is a hard
  contract, not an approximation — a unit test asserts these three channels are
  identically zero, so the placeholder nature is enforced in code, not just stated in a
  docstring.

Resampled to `T=64` via the existing `resample_time` from `stroke_dataset.py`, matching
how skeleton clips are already prepared — same interface a real IMU log will need to
satisfy.

### 3. Fusion model (`imu_fusion.py`: `IMUEncoder`, `FusedBeginnerExpertModel`)

- `IMUEncoder`: 3-layer 1D CNN over `(N, 6, T)` → global-average-pooled `(N, 32)` vector.
  Deliberately small (same reasoning as `STGCN`'s own size: little data, don't overbuild
  capacity).
- `FusedBeginnerExpertModel`: holds an `STGCN(num_classes=2)` skeleton branch and an
  `IMUEncoder`. Forward: `concat(skeleton_branch.extract_features(x_skel), imu_branch(x_imu))`
  → one `Linear(128 + 32, 2)` head.

### 4. Prototype training script (`train_beginner_expert_fusion_prototype.py`)

Runs the fused model on the existing subject-disjoint split, using
`synthetic_imu_from_skeleton` for the IMU input. Trains a few epochs, confirms the loop
completes and loss decreases. Saves to `results/imu_fusion_prototype/` — a **new,
separate** directory. Does not touch `results/beginner_expert_stgcn/` or any
cross-validation results. The script's own output/README callout explicitly states this
run is a code-path demonstration, not a benchmarked result, and must not be cited
alongside the 82.4% ± 3.8% number as if comparable.

### 5. Real-hardware data-collection protocol (documentation only)

For when hardware is acquired — no code, goes in README/future-work:

- **Hardware**: MPU6050 + ESP32 (built-in BLE/WiFi, cheap, locally available), mounted at
  the racket throat or handle.
- **Sampling rate**: ~100-200 Hz (a swing is ~0.3-0.5s; this gives 30-100 samples per
  swing, enough to resolve the contact spike).
- **Synchronization**: no shared clock between camera and IMU logger. Use a manual sync
  event — a sharp tap on the racket before each recorded take, which produces a visible
  spike in both the video and the accelerometer trace. Align streams by matching that
  spike, then compute the timestamp offset for the rest of that take.
- **Format**: one CSV/JSON log per clip (`timestamp, ax, ay, az, gx, gy, gz`), with the
  same `subject_id`/`skill_level`/`stroke` metadata fields the skeleton records already
  use, so a loader can join the two by clip ID.
- Once real logs exist, they resample to `T=64` the same way the synthetic generator's
  output already does — replacing `synthetic_imu_from_skeleton`'s call site is the only
  code change needed to move from prototype to real data.

## Data flow

```
skeleton JSON (existing)
  -> synthetic_imu_from_skeleton(): (T,6) proxy IMU signal [PROTOTYPE ONLY]
     (future: real MPU6050 CSV/JSON log, resampled the same way)
  -> IMUEncoder: (T,6) -> (32,) pooled feature
  -> STGCN.extract_features(): (T,17,2) skeleton -> (128,) pooled feature   [unchanged]
  -> concat -> Linear(160,2) -> beginner/expert logits
```

## Error handling

- `synthetic_imu_from_skeleton` uses only finite differences (subtraction), never
  division or windowing — degenerate short clips (`T` = 0, 1, or 2) fall out of numpy's
  slicing semantics as all-zero or minimal-length output, not a crash. No special-cased
  fallback is needed (verified: `quality.phases.dominant_wrist_index`, which this function
  calls, is already exercised at `T=0` by its own existing tests and has the same
  slicing-based safety).
- No new failure modes for real hardware data are handled here — that loader doesn't
  exist yet and is out of scope until hardware exists (see Out of scope).

## Testing

- `IMUEncoder`: shape `(N,6,T) -> (N,32)`, no NaNs.
- `synthetic_imu_from_skeleton`: deterministic (same input -> same output); channels 2, 3,
  4 (`accel_z, gyro_x, gyro_y`) are exactly zero for any input, enforcing the honesty
  contract in code; channels 0, 1, 5 are non-constant for a clip with actual wrist motion.
- `FusedBeginnerExpertModel`: forward pass produces `(N,2)` logits; a gradient-flow test
  confirms `loss.backward()` produces non-zero gradients on `IMUEncoder`'s parameters
  specifically (guards against silently building a fusion path whose output gets ignored
  by the final linear layer).
- Prototype training run: confirms the loop executes to completion and loss decreases
  over a few epochs on a small subset. Explicitly not a held-out accuracy claim — see
  Approach.

## Out of scope for this iteration

- Any real MPU6050 hardware, firmware, or data collection — hardware has not been
  acquired. §5 is a protocol for later, not implemented now.
- Any accuracy claim for the fusion architecture — cannot be made honestly on synthetic,
  skeleton-derived data. If real hardware data becomes available before the deadline, a
  follow-up iteration re-trains and cross-validates on it, following the same rigor as
  the existing v3 numbers, before any accuracy claim is made.
- Fusing IMU into the 6-way stroke classifier — scoped to the beginner/expert model only,
  per this session's decision. Reusable if wanted later since `IMUEncoder` and the
  refactored `STGCN.extract_features` are generic.
- Replacing or modifying the rule-based quality-scoring system (`quality/score.py`) — that
  system is unrelated to this fusion prototype and is left untouched.
