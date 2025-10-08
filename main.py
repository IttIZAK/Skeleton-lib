import base64
import cv2
import numpy as np
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mediapipe as mp
import math
from typing import Dict, Optional

app = FastAPI()
mp_pose = mp.solutions.pose

# ---------------------- Global states ----------------------
pose_states: Dict[str, Dict] = {}
hold_timers: Dict[str, Dict] = {}
reps_counts: Dict[str, Dict[str, int]] = {}
last_sent_time: Dict[str, float] = {}
frame_skip_config: Dict[str, int] = {}
selected_pose_per_client: Dict[str, Optional[str]] = {}

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
# ---------------------- Pose detech  ----------------------
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
    "Bodyweight Squat": is_squat,
    "Push-ups": is_pushup,
    "Plank": is_plank,
    "Sit-ups": is_situp,
    "Lunge (Forward Lunge)": is_forward_lunge,
    "Dead Bug": is_dead_bug,
    "Side Plank": is_side_plank,
    "Russian Twist": is_russian_twist,
    "Lying Leg Raises": is_lying_leg_raises
}

HOLD_POSES = {"Plank", "Side Plank"}
REPS_POSES = {"Bodyweight Squat", "Push-ups", "Sit-ups", "Lunge (Forward Lunge)",
              "Dead Bug", "Russian Twist", "Lying Leg Raises"}

# ---------------------- Feedback ----------------------
def feedback_squat(lm):
    R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    R_knee = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    angle = angle_between(R_hip, R_knee, R_ankle)

    if angle > 150:
        return "งอเข่าให้มากขึ้น"
    elif angle < 70:
        return "ยืดขึ้นมาอีกเล็กน้อย"
    return "ท่าถูกต้อง"

def feedback_pushup(lm):
    L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    L_el = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    L_wr = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
    R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])

    elbow_angle = angle_between(L_sh, L_el, L_wr)
    torso_diff = abs(((L_sh[1]+R_sh[1])/2) - ((L_hip[1]+R_hip[1])/2))

    if elbow_angle > 160:
        return "งอศอกลงอีกเพื่อทำ push-up ให้ลึก"
    elif elbow_angle < 50:
        return "ดันขึ้นมาอีกหน่อย"
    elif torso_diff > 0.1:
        return "เก็บสะโพกให้อยู่ในแนวเดียวกับไหล่"
    return "ท่าถูกต้อง"

def feedback_plank(lm):
    L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    L_ankle = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    torso_angle = angle_between(L_sh, L_hip, L_ankle)
    if torso_angle < 150:
        return "เก็บสะโพกให้อยู่ระดับเดียวกับลำตัว"
    return "ท่าถูกต้อง"

def feedback_situp(lm):
    L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    dist = abs(L_sh[1] - L_hip[1])
    if dist > 0.2:
        return "งอตัวขึ้นมาใกล้เข่ามากขึ้น"
    elif dist < 0.05:
        return "เอนตัวลงไปอีกนิด"
    return "ท่าถูกต้อง"

def feedback_forward_lunge(lm):
    R_knee = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    angle = angle_between(R_hip, R_knee, landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
    if angle > 150:
        return "งอเข่าหน้าให้มากขึ้น"
    return "ท่าถูกต้อง"

def feedback_dead_bug(lm):
    L_wr = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
    R_wr = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    L_an = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    R_an = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    avg_arm_y = (L_wr[1] + R_wr[1]) / 2
    avg_leg_y = (L_an[1] + R_an[1]) / 2
    if avg_arm_y > 0.8 or avg_leg_y > 0.8:
        return "ยกแขนหรือขาให้สูงขึ้น"
    return "ท่าถูกต้อง"

def feedback_side_plank(lm):
    L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    angle = angle_between(L_sh, L_hip, landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]))
    if angle < 150:
        return "ยกสะโพกขึ้นอีก"
    return "ท่าถูกต้อง"

def feedback_russian_twist(lm):
    nose = landmark_xy(lm[mp_pose.PoseLandmark.NOSE.value])
    L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    hip_center_x = (L_hip[0] + R_hip[0]) / 2
    if abs(nose[0] - hip_center_x) < 0.02:
        return "บิดลำตัวไปทางซ้ายหรือขวามากขึ้น"
    return "ท่าถูกต้อง"

def feedback_lying_leg_raises(lm):
    L_an = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    R_an = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    avg_an_y = (L_an[1] + R_an[1]) / 2
    if avg_an_y > 0.9:
        return "ยกขาขึ้นสูงกว่านี้"
    return "ท่าถูกต้อง"

# ---------------------- Mapping Feedback ----------------------
FEEDBACKS = {
    "Bodyweight Squat": feedback_squat,
    "Push-ups": feedback_pushup,
    "Plank": feedback_plank,
    "Sit-ups": feedback_situp,
    "Lunge (Forward Lunge)": feedback_forward_lunge,
    "Dead Bug": feedback_dead_bug,
    "Side Plank": feedback_side_plank,
    "Russian Twist": feedback_russian_twist,
    "Lying Leg Raises": feedback_lying_leg_raises
}



# ---------------------- Counter update ----------------------
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

# ---------------- WebSocket ----------------
@app.websocket("/ws/pose")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}_{int(time.time()*1000)}"
    print(f"[CONNECTED] {client_id}")

    frame_idx = 0
    frame_skip_config[client_id] = FRAME_SKIP
    selected_pose_per_client[client_id] = None

    with mp_pose.Pose(static_image_mode=False, model_complexity=0,
                      enable_segmentation=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose_detector:
        try:
            while True:
                data = await websocket.receive_text()
                frame_idx += 1

                # ---------- handle commands ----------
                if data.startswith("{"):
                    try:
                        cmd = json.loads(data)
                        if "frame_skip" in cmd:
                            frame_skip_config[client_id] = int(cmd["frame_skip"])
                        if "select_pose" in cmd:
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
                if (frame_idx % (frame_skip_config.get(client_id, FRAME_SKIP) + 1)) != 0:
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

                # prepare response template
                response = {
                    "pose": "N/A", "confidence": 0.0,
                    "reps": reps_counts.get(client_id, {}),
                    "holds": {},
                    "timestamp": ts,
                    "selected_pose": selected_pose_per_client.get(client_id)
                }

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    selected_pose = selected_pose_per_client.get(client_id)

                    if selected_pose and selected_pose in DETECTORS:
                        confidence = DETECTORS[selected_pose](landmarks)
                        update_counters(client_id, selected_pose, confidence, ts)

                        # ---------- update hold info ----------
                        user_holds = {}
                        for p, h in hold_timers.get(client_id, {}).items():
                            current_hold = round((ts - h["started_at"]) if h["started_at"] else 0.0, 2)
                            user_holds[p] = {
                                "current_hold": current_hold,
                                "best_hold": round(h["best"], 2)
                            }

                        # ---------- new: add advice ----------
                        advice = FEEDBACKS[selected_pose](landmarks) if selected_pose in FEEDBACKS else ""

                        response.update({
                            "pose": selected_pose,
                            "confidence": round(confidence, 3),
                            "reps": reps_counts.get(client_id, {}),
                            "holds": user_holds,
                            "advice": advice
                        })

                    else:
                        response["error"] = "no_pose_selected"
                else:
                    response["error"] = "no_pose"

                last_time = last_sent_time.get(client_id, 0)
                if ts - last_time >= DETECTION_COOLDOWN:
                    await websocket.send_text(json.dumps(response))
                    last_sent_time[client_id] = ts
                else:
                    await websocket.send_text(json.dumps({"status": "ok"}))

        except WebSocketDisconnect:
            print(f"[DISCONNECTED] {client_id}")
            for d in [pose_states, hold_timers, reps_counts,
                      last_sent_time, frame_skip_config, selected_pose_per_client]:
                d.pop(client_id, None)

# run: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000