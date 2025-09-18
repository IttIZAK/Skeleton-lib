import base64
import cv2
import numpy as np
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mediapipe as mp
import math
from typing import List, Dict, Optional

app = FastAPI()
mp_pose = mp.solutions.pose

# ---------------------- Global states ----------------------
pose_states: Dict[str, Dict] = {}
hold_timers: Dict[str, Dict] = {}
reps_counts: Dict[str, Dict[str, int]] = {}
last_detected: Dict[str, tuple] = {}
last_sent_time: Dict[str, float] = {}
frame_skip_config: Dict[str, int] = {}
selected_pose_per_client: Dict[str, Optional[str]] = {}   # ✅ เก็บท่าที่เลือก

# ---------------------- Config ----------------------
DETECTION_COOLDOWN = 0.5
FRAME_SKIP = 2
HOLD_MIN_CONFIDENCE = 0.6
POSE_START_THRESH = 0.6
POSE_END_THRESH = 0.4

# ---------------------- Helper functions ----------------------
def angle_between(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def landmark_xy(lm) -> tuple:
    return (lm.x, lm.y)
# ---------------------- Pose functions ----------------------
def is_squat(lm) -> float:
    try:
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        R_knee = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        L_knee = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
        L_ankle = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    except: 
        return 0.0
    knee_angle = (angle_between(R_hip,R_knee,R_ankle)+angle_between(L_hip,L_knee,L_ankle))/2
    hip_y = (R_hip[1]+L_hip[1])/2
    sh_y = (R_sh[1]+L_sh[1])/2
    angle_score = np.interp(knee_angle,[60,180],[1.0,0.0])
    hip_score = 1.0 if hip_y>sh_y+0.05 else 0.5
    return float(np.clip(angle_score*0.7+hip_score*0.3,0.0,1.0))

def is_pushup(lm) -> float:
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        L_el = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        L_wr = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    except: 
        return 0.0
    elbow_angle = angle_between(L_sh,L_el,L_wr)
    shoulder_y = (L_sh[1]+R_sh[1])/2
    hip_y = (L_hip[1]+R_hip[1])/2
    torso_flatness = 1.0 - np.clip(abs(shoulder_y-hip_y)*10,0,1)
    angle_score = np.interp(elbow_angle,[40,180],[1.0,0.0])
    return float(np.clip(angle_score*0.6+torso_flatness*0.4,0,1))

def is_plank(lm) -> float:
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_ankle = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    except: 
        return 0.0
    shoulder_y = (L_sh[1]+R_sh[1])/2
    hip_y = (L_hip[1]+R_hip[1])/2
    ankle_y = (L_ankle[1]+R_ankle[1])/2
    torso_flat = 1.0 - min(abs(shoulder_y-hip_y)/(abs(shoulder_y-ankle_y)*0.5+1e-6),1.0)
    hip_score = np.interp(hip_y-shoulder_y,[0,0.3],[1,0])
    return float(np.clip(torso_flat*0.6+hip_score*0.4,0,1))

def is_situp(lm) -> float:
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    except: 
        return 0.0
    dist = abs(L_sh[1]-L_hip[1])
    return float(np.clip(np.interp(dist,[0.02,0.25],[1,0]),0,1))

def is_forward_lunge(lm):
    try:
        R_knee = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_knee = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    except: return 0.0
    r_forward = R_knee[0]-R_hip[0]
    l_forward = L_knee[0]-L_hip[0]
    score = max(abs(r_forward),abs(l_forward))
    return float(np.clip(np.interp(score,[0.02,0.2],[0,1]),0,1))

def is_dead_bug(lm):
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_wr = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
        R_wr = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        L_an = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_an = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    except: return 0.0
    torso_flat = 1.0 - np.clip(abs(((L_sh[1]+R_sh[1])/2)-((L_hip[1]+R_hip[1])/2))*10,0,1)
    hip_y = (L_hip[1]+R_hip[1])/2
    limb_raise = max(
        0.0,
        np.interp(hip_y-L_wr[1],[0,0.3],[0,1]),
        np.interp(hip_y-R_wr[1],[0,0.3],[0,1]),
        np.interp(hip_y-L_an[1],[0,0.3],[0,1]),
        np.interp(hip_y-R_an[1],[0,0.3],[0,1])
    )
    return float(np.clip(torso_flat*0.6+limb_raise*0.4,0,1))

def is_side_plank(lm):
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    except: return 0.0
    hip_diff = abs(L_hip[1]-R_hip[1])
    sh_diff = abs(L_sh[1]-R_sh[1])
    score = np.clip(np.interp(hip_diff,[0.01,0.15],[0,1])*(1-np.interp(sh_diff,[0,0.15],[0,1])),0,1)
    return score

def is_russian_twist(lm):
    try:
        nose = landmark_xy(lm[mp_pose.PoseLandmark.NOSE.value])
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    except: return 0.0
    twist = abs((L_sh[0]+R_sh[0])/2-(L_hip[0]+R_hip[0])/2)
    return float(np.clip(np.interp(twist,[0.01,0.12],[0,1]),0,1))

def is_lying_leg_raises(lm):
    try:
        L_an = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_an = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    except: return 0.0
    avg_an = (L_an[1]+R_an[1])/2
    avg_hip = (L_hip[1]+R_hip[1])/2
    diff = avg_hip-avg_an
    return float(np.clip(np.interp(diff,[0.02,0.3],[0,1]),0,1))

# ---------------------- Mapping ----------------------
DETECTORS = {
    "squat": is_squat,
    "pushup": is_pushup,
    "plank": is_plank,
    "situp": is_situp,
    "forward_lunge": is_forward_lunge,
    "dead_bug": is_dead_bug,
    "side_plank": is_side_plank,
    "russian_twist": is_russian_twist,
    "lying_leg_raises": is_lying_leg_raises
}

HOLD_POSES = {"plank", "side_plank"}
REPS_POSES = {"squat", "pushup", "situp", "forward_lunge",
              "dead_bug", "russian_twist", "lying_leg_raises"}

def update_counters(client_id, pose_name, confidence, ts):
    reps_counts.setdefault(client_id, {})
    pose_states.setdefault(client_id, {})
    hold_timers.setdefault(client_id, {})

    if pose_name in HOLD_POSES:
        ht = hold_timers[client_id].setdefault(pose_name, {"started_at": None, "best": 0.0})
        if confidence >= HOLD_MIN_CONFIDENCE:
            if ht["started_at"] is None:
                ht["started_at"] = ts
            hold_duration = ts - ht["started_at"]
            if hold_duration > ht["best"]:
                ht["best"] = hold_duration
        else:
            if ht["started_at"] is not None:
                hold_duration = ts - ht["started_at"]
                ht["started_at"] = None
                if hold_duration > ht["best"]:
                    ht["best"] = hold_duration
    else:
        state = pose_states[client_id].setdefault(pose_name, {"in_pose": False})
        if confidence >= POSE_START_THRESH and not state["in_pose"]:
            state["in_pose"] = True
        elif confidence <= POSE_END_THRESH and state["in_pose"]:
            reps_counts[client_id].setdefault(pose_name, 0)
            reps_counts[client_id][pose_name] += 1
            state["in_pose"] = False

    last_detected[client_id] = (pose_name, confidence, ts)

# ---------------- WebSocket ----------------
@app.websocket("/ws/pose")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}_{int(time.time()*1000)}"
    print(f"[CONNECTED] {client_id}")

    frame_idx, frame_skip = 0, FRAME_SKIP
    frame_skip_config[client_id] = frame_skip
    pose_states.setdefault(client_id, {})
    hold_timers.setdefault(client_id, {})
    reps_counts.setdefault(client_id, {})
    last_detected.setdefault(client_id, (None, 0.0, 0.0))
    last_sent_time.setdefault(client_id, 0.0)
    selected_pose_per_client[client_id] = None  # ✅ เริ่มต้นยังไม่ได้เลือกท่า

    with mp_pose.Pose(static_image_mode=False, model_complexity=0,
                      enable_segmentation=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose_detector:
        try:
            while True:
                data = await websocket.receive_text()
                frame_idx += 1

                # ---------- handle config/commands ----------
                if data.startswith("{"):  
                    try:
                        cmd = json.loads(data)
                        if "frame_skip" in cmd:
                            frame_skip_config[client_id] = int(cmd["frame_skip"])
                        if "save_detected" in cmd:
                            global FRAME_SAVE_DETECTED
                            FRAME_SAVE_DETECTED = bool(cmd["save_detected"])
                        if "select_pose" in cmd:   # ✅ รับท่าจาก Flutter
                            pose_name = cmd["select_pose"]
                            if pose_name in DETECTORS:
                                selected_pose_per_client[client_id] = pose_name
                                await websocket.send_text(json.dumps({
                                    "status": "pose_selected",
                                    "pose": pose_name
                                }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "status": "error",
                                    "detail": f"Unknown pose: {pose_name}"
                                }))
                    except:
                        pass
                    continue

                # ---------- frame skip ----------
                if (frame_idx % (frame_skip + 1)) != 0:
                    await websocket.send_text(json.dumps({"status": "ok"}))
                    continue

                # ---------- decode frame ----------
                try:
                    frame = cv2.imdecode(
                        np.frombuffer(base64.b64decode(data), np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        raise ValueError("frame decode failed")
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": "decode_failed", "detail": str(e)}))
                    continue

                ts = time.time()
                results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                response = {
                    "pose": "N/A", "confidence": 0.0,
                    "reps": {}, "holds": {}, "timestamp": ts,
                    "selected_pose": selected_pose_per_client.get(client_id)
                }

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    selected_pose = selected_pose_per_client.get(client_id)

                    if selected_pose and selected_pose in DETECTORS:
                        # ✅ ตรวจจับเฉพาะท่าที่เลือก
                        confidence = DETECTORS[selected_pose](landmarks)
                        update_counters(client_id, selected_pose, confidence, ts)

                        user_reps = reps_counts.get(client_id, {})
                        user_holds = {
                            p: {"current_hold": round((ts - h["started_at"]) if h["started_at"] else 0.0, 2),
                                "best_hold": round(h["best"], 2)}
                            for p, h in hold_timers.get(client_id, {}).items()
                        }

                        response.update({
                            "pose": selected_pose,
                            "confidence": round(confidence, 3),
                            "reps": user_reps,
                            "holds": user_holds
                        })
                    else:
                        response["error"] = "no_pose_selected"
                else:
                    response["error"] = "no_pose"

                if ts - last_sent_time[client_id] >= DETECTION_COOLDOWN:
                    await websocket.send_text(json.dumps(response))
                    last_sent_time[client_id] = ts
                else:
                    await websocket.send_text(json.dumps({"status": "ok"}))

        except WebSocketDisconnect:
            print(f"[DISCONNECTED] {client_id}")
            for d in [pose_states, hold_timers, reps_counts,
                      last_detected, last_sent_time, frame_skip_config,
                      selected_pose_per_client]:
                d.pop(client_id, None)

# run: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000