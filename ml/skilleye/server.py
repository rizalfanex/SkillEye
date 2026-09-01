import base64
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rtmlib import Body, Hand

from stroke_dataset import STROKE_CLASSES, resample_time, add_velocity
from stgcn_model import STGCN, COCO17_EDGES
from skeleton_pipeline import clean_clip
from quality.score import score_clip, score_clip_correlated
from quality.phases import contact_window_for
from quality.skill_rules import (
    check_volley_swing_effort, evaluate_backhand_volley_skill_rules, g_to_mps2,
)

HARDWARE_CLIENT_DIR = Path(__file__).resolve().parents[2] / "hardware" / "client"
if str(HARDWARE_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_CLIENT_DIR))
from imu_client import IMUStream, StreamStats, iter_rows, load_config

HARDWARE_CLIENT_DIR = Path(__file__).resolve().parents[2] / "hardware" / "client"
if str(HARDWARE_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_CLIENT_DIR))
from imu_client import IMUStream, StreamStats, iter_rows, load_config

APP_DIR = Path(__file__).parent.resolve()
TEMPLATES_PATH = str(APP_DIR / "../results/quality_templates/templates.json")
STROKE_MODEL_PATH = str(APP_DIR / "../results/stroke_classifier_v2/best_model.pt")

app = FastAPI(title="SkillEye AI Analysis Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model caches
pose_body = None
hand_body = None
stroke_model = None
templates = None
covariance = None

STROKE_DISPLAY_MAP = {
    "forehand": "正手拍 (Forehand)",
    "backhand": "反手拍 (Backhand)",
    "serve": "發球 (Serve)",
    "smash": "高壓殺球 (Smash)",
    "forehand_volley": "正手截擊 (Forehand Volley)",
    "backhand_volley": "反手截擊 (Backhand Volley)",
}

IMU_CONFIG = load_config()
# Where paired browser recordings are written.
#
# This directory lives inside the repo, which is fine for git (it is ignored)
# but not for a dev server that watches the workspace. VS Code Live Server
# reloads the browser on any file change, and a paired recording creates its
# CSV the moment the session starts -- so the page reloads mid-take, the
# recording dies, and the analysis page resets to its defaults. It looks exactly
# like "recording starts and stops by itself", and only in Camera + racket mode,
# because camera-only writes no file. .vscode/settings.json now tells Live
# Server to ignore this path; set SKILLEYE_IMU_RECORDINGS_DIR to move the
# recordings out of the watched tree entirely for any other setup.
IMU_RECORDINGS_DIR = Path(
    os.environ.get("SKILLEYE_IMU_RECORDINGS_DIR")
    or (APP_DIR / "../results/imu_recordings")
).resolve()

# Stroke-detection gate, measured against the project's own recordings
# (hardware/client/newresult/, fh_batch01 + bh_batch01).
#
# The earlier values (2.0 g / 150 deg/s) did not separate a stroke from ordinary
# racket handling. Across both takes, the peaks reached OUTSIDE any stroke were
# 1.86 g and 404 deg/s -- the gyro threshold in particular sat far below the
# noise floor of simply holding and repositioning the racket, and 45% of
# stroke-free 3-second windows passed the gate.
#
# Real strokes in the same recordings peaked at 5.44-11.21 g and 705-977 deg/s,
# so there is a wide clean gap. These thresholds sit inside it: on the same
# sliding-window test they give 0% false accepts and 100% detection.
IMU_STROKE_ACCEL_THRESHOLD_G = 3.5
IMU_STROKE_GYRO_THRESHOLD_DPS = 500.0

# A browser that closes mid-recording never calls /imu/session/stop. Without a
# deadline the stale session would own the global forever and every later start
# would 409, which is a bricked feature until the server restarts.
IMU_SESSION_MAX_SECONDS = 180.0

imu_session = None
imu_session_lock = threading.Lock()


def pick_device():
    """Return 'cuda' if a working GPU backend exists, else 'cpu'."""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class FramesPayload(BaseModel):
    frames: list[str]  # base64-encoded JPEG frames
    fps: float = 10.0
    stroke: str | None = None
    # Optional fields the master frontend (SkillEye-master) sends so we can echo
    # them back and let the page render its "model extras" notes truthfully.
    scoring_mode: str | None = None
    with_ball: bool | None = None
    contact_time_s: float | None = None
    peak_accel_g: float | None = None
    # "independent" (v1) or "correlated" (Module A). Same names the demo's
    # sidebar toggle uses, so the two front-ends cannot mean different things.
    scoring_mode: str = "independent"
    # Seconds from the first submitted frame to the instant the racket IMU says
    # contact happened. Optional: without it the existing wrist-speed heuristic
    # is used exactly as before.
    contact_time_s: float | None = None
    # Peak dominant-hand acceleration for this stroke, in g, straight off the
    # accelerometer. Feeds the published volley rule in quality/skill_rules.py.
    peak_accel_g: float | None = None
    # Whether the player was actually striking a ball. Declared by the user --
    # a 200 Hz stream cannot tell (see impactSignature in skilleye-imu.js).
    # Gates the rules that only mean something on a real impact.
    with_ball: bool = False


def load_resources():
    global pose_body, hand_body, stroke_model, templates, covariance
    device = pick_device()

    if pose_body is None:
        print("Loading RTMPose estimator (device={})...".format(device))
        pose_body = Body(mode="lightweight", to_openpose=False, backend="onnxruntime", device=device)

    if hand_body is None:
        print("Loading RTMPose hand estimator (device={})...".format(device))
        hand_body = Hand(mode="lightweight", to_openpose=False, backend="onnxruntime", device=device)
    
    if stroke_model is None:
        print("Loading ST-GCN stroke model (device={})...".format(device))
        torch_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        stroke_model = STGCN(num_classes=len(STROKE_CLASSES))
        stroke_model.load_state_dict(torch.load(STROKE_MODEL_PATH, map_location=torch_device))
        stroke_model.to(torch_device)
        stroke_model.eval()

    if templates is None:
        print("Loading expert biomechanics templates...")
        with open(TEMPLATES_PATH) as f:
            data = json.load(f)
        templates = data["templates"]
        # Module A needs the per-phase covariance blocks that
        # build_expert_templates.compute_covariance_templates() writes alongside
        # the independent templates. Absent in an older templates.json, in which
        # case scoring silently stays on the independent path.
        covariance = data.get("covariance", {})


@app.on_event("startup")
def startup_event():
    load_resources()


def pose_frames(frames, step=1):
    """Run RTMPose on a list of BGR frames. step>1 subsamples frames to speed up
    full-video analysis while still capturing the swing trajectory."""
    per_frame_kpts, per_frame_scores = [], []
    for idx, frame in enumerate(frames):
        if idx % step != 0:
            continue
        keypoints, scores = pose_body(frame)
        per_frame_kpts.append(np.asarray(keypoints))
        per_frame_scores.append(np.asarray(scores))
    return per_frame_kpts, per_frame_scores


def check_motion_intensity(kpts, threshold=10.0):
    """Checks if the clip has significant sudden acceleration in arm joints (characteristic of a stroke)."""
    # Key joints for tennis stroke: Shoulder (5,6), Elbow (7,8), Wrist (9,10)
    critical_indices = [5, 6, 7, 8, 9, 10]
    kpts_critical = kpts[:, critical_indices, :]
    
    # Calculate acceleration (second derivative of position)
    vel = np.diff(kpts_critical, axis=0) # (T-1, V, 2)
    acc = np.diff(vel, axis=0)           # (T-2, V, 2)
    
    # Peak acceleration magnitude across these joints
    acc_mag = np.linalg.norm(acc, axis=2) # (T-2, V)
    peak_acc = np.max(acc_mag)
    return peak_acc > threshold


def analyze_kpts(per_frame_kpts, per_frame_scores, override_stroke=None,
                contact_index=None, peak_accel_g=None, scoring_mode="independent",
                with_ball=False, rows_per_second=None):
    # ... (前略)
        
    # NEW: Check if motion intensity is significant (explosive acceleration)
    try:
        if not check_motion_intensity(np.asarray(per_frame_kpts)):
            raise HTTPException(
                status_code=400,
                detail="動作幅度或爆發力不足，請確認是否為標準網球揮拍動作。"
            )
    except HTTPException:
        raise
    except Exception:
        # Not enough frames or invalid keypoints -> skip motion-intensity gate
        pass

    # Clean the clip (single-subject tracking, resample to fixed length)
    kpts_arr, scores_arr, ok = clean_clip(per_frame_kpts, per_frame_scores)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Primary subject visible in too few frames. Please use a clearer video of a single player."
        )

    # The classifier runs either way. When the user declared the stroke it does
    # not decide the scoring, but it is the only thing that can notice the clip
    # holds no recognisable stroke at all -- and a declared stroke used to skip
    # it entirely, so live camera-only scored whatever it was given, including
    # a clip with no swing in it, and reported confidence 1.0 for a model that
    # never ran.
    model_stroke = None
    model_confidence = None
    try:
        resampled = resample_time(kpts_arr, 64)
        with_velocity = add_velocity(resampled)
        tensor = torch.from_numpy(with_velocity.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        device = next(stroke_model.parameters()).device
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = stroke_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        model_stroke = STROKE_CLASSES[pred_idx]
        model_confidence = float(probs[pred_idx])
    except Exception:
        model_stroke, model_confidence = None, None

    if override_stroke:
        pred_stroke = override_stroke
        pred_confidence = model_confidence if model_confidence is not None else 1.0
    else:
        if model_stroke is None:
            raise HTTPException(status_code=500, detail="Stroke classifier failed on this clip.")
        pred_stroke = model_stroke
        pred_confidence = model_confidence

    # Score against expert biomechanics templates
    # Module A (correlated z-score, README 2.8) scores each joint against its
    # expected value GIVEN what the other joints are doing in that phase,
    # instead of against its own template alone. Same output shape as the
    # independent scorer, so nothing downstream changes. Mirrors app.py's
    # sidebar toggle, including its fallback: a templates.json without a
    # covariance block for this stroke drops back to independent rather than
    # failing, and says so.
    # Rows either side of contact, sized so the contact phase covers the same
    # slice of real time whichever endpoint produced these rows.
    contact_window = contact_window_for(rows_per_second)

    scoring_used = "independent"
    scoring_note = None
    if scoring_mode == "correlated":
        if covariance and pred_stroke in covariance and covariance[pred_stroke]:
            result = score_clip_correlated(kpts_arr, pred_stroke, covariance,
                                           contact=contact_index,
                                           contact_window=contact_window)
            scoring_used = "correlated"
        else:
            result = score_clip(kpts_arr, pred_stroke, templates, contact=contact_index,
                                contact_window=contact_window)
            scoring_note = (
                f"No covariance data for '{pred_stroke}' in this templates.json "
                "(rebuild with build_expert_templates.py) -- fell back to independent "
                "scoring for this clip.")
    else:
        result = score_clip(kpts_arr, pred_stroke, templates, contact=contact_index,
                            contact_window=contact_window)

    # NEW: Check stroke prediction confidence threshold (Strict: 0.85)
    STROKE_CONFIDENCE_THRESHOLD = 0.85 
    if not override_stroke and pred_confidence < STROKE_CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"動作不明確 (信心值 {pred_confidence:.2f})，請確認是否為揮拍動作。"
        )


    # Skill-level rules from the team's own quality/skill_rules.py. These encode
    # findings from papers that compare skilled against less-skilled players
    # directly, which the expert-only z-score above cannot express. Until now
    # they were reachable only from the Streamlit demo (app.py), so the website
    # never saw them.
    skill_flags = []
    if pred_stroke == "backhand_volley":
        try:
            skill_flags = evaluate_backhand_volley_skill_rules(
                kpts_arr, contact=contact_index,
                contact_window=contact_window)["flags"]
        except Exception:
            skill_flags = []

    volley_effort = None
    if peak_accel_g is not None and pred_stroke in ("backhand_volley", "forehand_volley"):
        volley_effort = check_volley_swing_effort(
            g_to_mps2(float(peak_accel_g)), with_ball=with_ball)

    # Declared stroke: score it as asked, but say so when the model disagrees or
    # is unsure. Silently scoring an unrecognisable clip against a stroke the
    # user picked is how a score comes back for a swing that never happened.
    stroke_warning = None
    if override_stroke and model_stroke is not None:
        if model_confidence is not None and model_confidence < STROKE_CONFIDENCE_THRESHOLD:
            stroke_warning = (
                f"動作不明確（模型信心值 {model_confidence:.2f} < {STROKE_CONFIDENCE_THRESHOLD}）。"
                f"仍以你選擇的「{STROKE_DISPLAY_MAP.get(override_stroke, override_stroke)}」計分，"
                "但這段影像可能不是一次完整的揮拍，分數僅供參考。")
        elif model_stroke != override_stroke:
            stroke_warning = (
                f"模型判斷這是「{STROKE_DISPLAY_MAP.get(model_stroke, model_stroke)}」"
                f"（信心值 {model_confidence:.2f}），與你選擇的"
                f"「{STROKE_DISPLAY_MAP.get(override_stroke, override_stroke)}」不同。"
                "仍以你的選擇計分——若選錯了，會拿去和錯誤的專家範本比較。")

    return {
        "overall_score": float(result["overall_score"]),
        "predicted_stroke": pred_stroke,
        "predicted_stroke_display": STROKE_DISPLAY_MAP.get(pred_stroke, pred_stroke),
        "predicted_confidence": pred_confidence,
        "contact_index": contact_index,
        "contact_source": "racket_imu" if contact_index is not None else "wrist_speed",
        "scoring_mode": scoring_used,
        "scoring_note": scoring_note,
        "stroke_source": "declared" if override_stroke else "model",
        "model_stroke": model_stroke,
        "model_stroke_display": STROKE_DISPLAY_MAP.get(model_stroke, model_stroke) if model_stroke else None,
        "model_confidence": model_confidence,
        "stroke_warning": stroke_warning,
        "contact_window": contact_window,
        "rows_per_second": rows_per_second,
        "skill_flags": skill_flags,
        "volley_effort": volley_effort,
        "with_ball": with_ball,
        "table": result["table"],
        "suggestions": result["suggestions"],
    }


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    stroke: str | None = Query(None),
    contact_time_s: float | None = Query(None),
    peak_accel_g: float | None = Query(None),
    scoring_mode: str = Query("independent"),
    with_ball: bool = Query(False),
):
    load_resources()

    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        frames, video_fps = read_video_frames(tmp_path)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not decode frames.")
        # Subsample to 1 of every 3 frames for speed (GPU: ~0.12s/frame)
        step = 3
        per_frame_kpts, per_frame_scores = pose_frames(frames, step=step)

        # Contact arrives as a time from the start of the video; pose rows exist
        # only every `step` frames, so divide by the subsample factor too.
        contact_index = None
        if contact_time_s is not None:
            cap_fps = video_fps or 30.0
            contact_index = int(round(contact_time_s * cap_fps / step))
            contact_index = max(0, min(len(per_frame_kpts) - 1, contact_index))

        # Rows come out of pose_frames at video_fps / step.
        rows_per_second = (video_fps or 30.0) / step

        return analyze_kpts(per_frame_kpts, per_frame_scores,
                            override_stroke=stroke,
                            contact_index=contact_index, peak_accel_g=peak_accel_g,
                            scoring_mode=scoring_mode, with_ball=with_ball,
                            rows_per_second=rows_per_second)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)


def _imu_connection():
    return IMUStream(
        IMU_CONFIG.get("host", "192.168.4.1"),
        int(IMU_CONFIG.get("port", 3333)),
        timeout=10.0,
    )


def _probe_imu():
    """Connect and read the first few lines to find out what we actually got.

    A bare TCP connect is not enough to say the device is usable. The firmware
    is single-client (firmware.ino acceptClient): when a second program is
    already streaming, the ESP32 still ACCEPTS the connection, writes
    '# busy: another client is already streaming', and hangs up. A connect-only
    check therefore reports a healthy sensor while it is impossible to record --
    including whenever imu_client.py is left running, which is the normal way
    the team uses the device.
    """
    with _imu_connection() as stream:
        for i, line in enumerate(stream.lines()):
            if "busy" in line.lower():
                return {"connected": True, "streaming": False, "reason": line.strip()}
            if line.startswith("seq,"):          # column header -> stream is live
                return {"connected": True, "streaming": True}
            if i > 12:                            # header block is short; stop reading
                return {"connected": True, "streaming": True}
    return {"connected": True, "streaming": False, "reason": "device closed the connection"}


@app.get("/imu/status")
async def imu_status():
    """Report whether the ESP32 IMU is reachable AND free to stream."""
    with imu_session_lock:
        busy_locally = imu_session is not None
    if busy_locally:
        # Do not open a competing connection while our own recording is running.
        return {"connected": True, "streaming": False, "reason": "a recording is already active"}
    try:
        return await asyncio.to_thread(_probe_imu)
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        return {"connected": False, "streaming": False, "error": str(exc)}


def _record_imu(seconds, output_path):
    stats = StreamStats()
    started = time.monotonic()
    with _imu_connection() as stream, open(output_path, "w", encoding="utf-8", newline="") as fh:
        for line in stream.lines():
            fh.write(line + "\n")
            for _ in iter_rows([line], stats):
                pass
            if time.monotonic() - started >= seconds:
                break
    return stats


def _imu_session_worker(session):
    started = time.monotonic()
    try:
        with _imu_connection() as stream:
            session["connected"] = True
            session["ready"].set()
            with open(session["output_path"], "w", encoding="utf-8", newline="") as fh:
                for line in stream.lines():
                    if session["stop"].is_set():
                        break
                    # Own deadline, so a browser that vanished cannot leave this
                    # thread streaming to disk for the rest of the server's life.
                    if time.monotonic() - started > IMU_SESSION_MAX_SECONDS:
                        session["timed_out"] = True
                        break
                    fh.write(line + "\n")
                    for _, t_us, values in iter_rows([line], session["stats"]):
                        # Anchor the device's monotonic clock to the recording
                        # computer's wall clock on the very first sample. This is
                        # the same trick sync_recorder.py writes into
                        # *_alignment.json, and it makes live alignment exact
                        # instead of something to estimate from content.
                        if session["first_t_us"] is None:
                            session["first_t_us"] = t_us
                            session["first_wall_us"] = int(time.time() * 1_000_000)
                        accel = (values[0] ** 2 + values[1] ** 2 + values[2] ** 2) ** 0.5
                        gyro = (values[3] ** 2 + values[4] ** 2 + values[5] ** 2) ** 0.5
                        session["peak_accel_g"] = max(session["peak_accel_g"], accel)
                        session["peak_gyro_dps"] = max(session["peak_gyro_dps"], gyro)
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        session["error"] = str(exc)
    finally:
        session["finished"].set()
        session["ready"].set()


@app.post("/imu/session/start")
async def start_imu_session():
    """Start collecting IMU data for the browser's paired recording."""
    global imu_session
    with imu_session_lock:
        if imu_session is not None:
            # Only refuse if the previous recording is genuinely still running.
            # A browser that closed mid-take never calls stop, and treating that
            # stale entry as "active" forever would brick every later recording
            # until the server restarted.
            previous = imu_session
            if previous["finished"].is_set() or not previous["thread"].is_alive():
                imu_session = None
            else:
                previous["stop"].set()
                if previous["finished"].wait(2.0):
                    imu_session = None
                else:
                    raise HTTPException(
                        status_code=409, detail="An IMU recording is already active")
        IMU_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = IMU_RECORDINGS_DIR / f"browser_{int(time.time() * 1_000_000)}.csv"
        session = {
            "stop": threading.Event(), "ready": threading.Event(),
            "finished": threading.Event(),
            "stats": StreamStats(), "output_path": output_path,
            "connected": False, "error": None, "timed_out": False,
            "peak_accel_g": 0.0, "peak_gyro_dps": 0.0,
            "first_t_us": None, "first_wall_us": None,
        }
        imu_session = session
        session["thread"] = threading.Thread(
            target=_imu_session_worker, args=(session,), daemon=True)
        session["thread"].start()

    await asyncio.to_thread(session["ready"].wait, 10.0)
    if session["error"]:
        with imu_session_lock:
            imu_session = None
        raise HTTPException(status_code=503, detail=f"IMU connection failed: {session['error']}")
    return {"connected": session["connected"]}


@app.post("/imu/session/stop")
async def stop_imu_session():
    """Stop the paired IMU recording and report whether a stroke was detected."""
    global imu_session
    with imu_session_lock:
        session = imu_session
        imu_session = None
    if session is None:
        raise HTTPException(status_code=404, detail="No active IMU recording")

    session["stop"].set()
    await asyncio.to_thread(session["thread"].join, 12.0)
    stroke_detected = (
        session["peak_accel_g"] >= IMU_STROKE_ACCEL_THRESHOLD_G
        or session["peak_gyro_dps"] >= IMU_STROKE_GYRO_THRESHOLD_DPS
    )

    # Hand the recording itself back, not just the yes/no gate. The browser's
    # analysis engine (website/js/skilleye-imu.js) derives racket-face angle,
    # swing tempo and per-stroke physics from this CSV -- without it the file
    # would sit on disk unused and "with sensor" would still produce the same
    # report as "camera only".
    csv_text = None
    try:
        if session["output_path"].exists():
            csv_text = session["output_path"].read_text(encoding="utf-8")
    except OSError:
        csv_text = None

    return {
        "stroke_detected": stroke_detected,
        "peak_accel_g": session["peak_accel_g"],
        "peak_gyro_dps": session["peak_gyro_dps"],
        "rows": session["stats"].rows,
        "missing": session["stats"].missing,
        "recording": str(session["output_path"].relative_to(APP_DIR.parent.parent)),
        "timed_out": session["timed_out"],
        # Clock anchor: IMU sample t_us maps to wall-clock microseconds as
        #   wall = first_wall_us + (t_us - first_t_us)
        "first_t_us": session["first_t_us"],
        "first_wall_us": session["first_wall_us"],
        "accel_threshold_g": IMU_STROKE_ACCEL_THRESHOLD_G,
        "gyro_threshold_dps": IMU_STROKE_GYRO_THRESHOLD_DPS,
        "csv": csv_text,
    }


@app.post("/imu/record")
async def record_imu(
    seconds: float = Query(10.0, gt=0, le=300),
    filename: str = Query("imu_recording.csv"),
):
    """Record raw ESP32 IMU CSV data for a bounded duration."""
    safe_filename = Path(filename).name
    if not safe_filename.lower().endswith(".csv"):
        safe_filename += ".csv"
    IMU_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = IMU_RECORDINGS_DIR / safe_filename
    try:
        stats = await asyncio.to_thread(_record_imu, seconds, output_path)
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail=f"IMU connection failed: {exc}") from exc

    return {
        "recording": str(output_path.relative_to(APP_DIR.parent.parent)),
        "rows": stats.rows,
        "missing": stats.missing,
        "duration_s": stats.duration_s,
        "measured_hz": stats.measured_hz,
        "metadata": stats.metadata,
    }


@app.post("/analyze_frame")
async def analyze_frame(image: UploadFile = File(...)):
    load_resources()
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(400, "Decode error")

    keypoints, scores = pose_body(frame)
    kpts = np.asarray(keypoints); sc = np.asarray(scores)
    persons = []
    if kpts.ndim == 3 and kpts.shape[0] > 0:
        for p in range(kpts.shape[0]):
            persons.append([
                {"x": float(kpts[p, j, 0]), "y": float(kpts[p, j, 1]), "score": float(sc[p, j])}
                for j in range(kpts.shape[1])
            ])
    return {"persons": persons}


# MediaPipe/RTMPose hand landmark indices
_HAND_FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_HAND_FINGER_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18} # Use MCP (2) for thumb


def _finger_extended(kpts, tip_idx, pip_idx, min_score=0.3):
    """A finger counts as extended when its tip is clearly farther from the
    wrist than its PIP joint -- i.e. the finger is straight rather than bent."""
    if kpts[tip_idx, 2] < min_score or kpts[pip_idx, 2] < min_score:
        return False
    wrist = kpts[0, :2]
    d_tip = float(np.linalg.norm(kpts[tip_idx, :2] - wrist))
    d_pip = float(np.linalg.norm(kpts[pip_idx, :2] - wrist))
    # Reduced multiplier to 1.15 for more natural and robust open-palm detection
    return d_tip > d_pip * 1.15


def is_five_gesture(kpts):
    """True when all five fingers of a hand are extended (open palm)."""
    return all(
        _finger_extended(kpts, _HAND_FINGER_TIPS[f], _HAND_FINGER_PIPS[f])
        for f in _HAND_FINGER_TIPS
    )


@app.post("/detect_gesture")
async def detect_gesture(image: UploadFile = File(...)):
    load_resources()
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(400, "Decode error")

    keypoints, scores = hand_body(frame)
    kpts = np.asarray(keypoints); sc = np.asarray(scores)
    hands = []
    if kpts.ndim == 3 and kpts.shape[0] > 0:
        for p in range(kpts.shape[0]):
            mean_score = float(sc[p].mean())
            if mean_score < 0.6:  # Reject low-confidence hands / false positives
                continue
            hand_kpts = np.column_stack([kpts[p], sc[p]])
            hands.append({
                "five": bool(is_five_gesture(hand_kpts)),
                "score": mean_score,
            })
    return {"hands": hands, "five": any(h["five"] for h in hands)}


@app.post("/analyze_frames")
async def analyze_frames(payload: FramesPayload):
    load_resources()
    frames = []
    for b64 in payload.frames:
        raw = base64.b64decode(b64.split(",")[-1])
        arr = np.frombuffer(raw, np.uint8)
        f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if f is not None: frames.append(f)
    if not frames: raise HTTPException(400, "No frames")
    per_frame_kpts, per_frame_scores = pose_frames(frames)

    # The browser sends contact as a time, not an index, because only the server
    # knows how the frames it received map onto pose rows (pose_frames may
    # subsample). One conversion, in one place.
    contact_index = None
    if payload.contact_time_s is not None and payload.fps > 0:
        contact_index = int(round(payload.contact_time_s * payload.fps))
        contact_index = max(0, min(len(per_frame_kpts) - 1, contact_index))

    # step=1 above, so one pose row per frame the browser actually captured --
    # which is the rate it measured and sent, not any nominal camera frame rate.
    return analyze_kpts(
        per_frame_kpts, per_frame_scores,
        override_stroke=payload.stroke,
        contact_index=contact_index,
        peak_accel_g=payload.peak_accel_g,
        scoring_mode=payload.scoring_mode,
        with_ball=payload.with_ball,
        rows_per_second=(payload.fps if payload.fps and payload.fps > 0 else None),
    )


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames, fps


# Serve the frontend (website/analysis.html, etc.) from the same server URL, so
# opening e.g. http://127.0.0.1:8000/SkillEye/website/analysis.html just works and
# the page's API_BASE (127.0.0.1:8000) lines up with the API. WEB_ROOT is the
# SkillEye repo root (two levels above ml/skilleye).
WEB_ROOT = APP_DIR.parent.parent
if WEB_ROOT.exists():
    app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)