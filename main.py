"""
Pose Detection API - Production Ready
ระบบตรวจจับท่าออกกำลังกายด้วย MediaPipe
รองรับ 9 ท่าพร้อม Real-time Feedback
"""

import base64
import cv2
import numpy as np
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pose Detection API",
    description="Real-time exercise pose detection and counting",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose

# ==================== Data Classes ====================
@dataclass
class PoseState:
    """สถานะของท่าออกกำลังกาย"""
    in_pose: bool = False
    last_detection_time: float = 0.0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=5))

@dataclass
class HoldTimer:
    """Timer สำหรับท่าที่ต้องถือท่า"""
    started_at: Optional[float] = None
    best: float = 0.0
    current_streak: int = 0
    
@dataclass
class ClientState:
    """สถานะของแต่ละ client"""
    pose_states: Dict[str, PoseState] = field(default_factory=dict)
    hold_timers: Dict[str, HoldTimer] = field(default_factory=dict)
    reps_counts: Dict[str, int] = field(default_factory=dict)
    selected_pose: Optional[str] = None
    frame_skip: int = 2
    last_feedback_time: float = 0.0
    last_advice: str = ""

# ==================== Global States ====================
client_states: Dict[str, ClientState] = {}

# ==================== Configuration ====================
class Config:
    """การตั้งค่าระบบ"""
    DETECTION_COOLDOWN = 0.3
    FRAME_SKIP = 2
    HOLD_MIN_CONFIDENCE = 0.70
    POSE_START_THRESH = 0.65
    POSE_END_THRESH = 0.45
    FEEDBACK_INTERVAL = 1.5
    MIN_CONFIDENCE_HISTORY = 3
    MODEL_COMPLEXITY = 1
    MIN_DETECTION_CONFIDENCE = 0.6
    MIN_TRACKING_CONFIDENCE = 0.6

config = Config()

# ==================== Helper Functions ====================
def angle_between(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """คำนวณมุมระหว่างจุด 3 จุด (องศา)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    
    cosang = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def landmark_xy(lm) -> Tuple[float, float]:
    """ดึงค่า x, y จาก landmark"""
    return (lm.x, lm.y)

def check_visibility(lm, threshold: float = 0.5) -> bool:
    """ตรวจสอบว่า landmark มองเห็นชัดเจน"""
    return lm.visibility > threshold if hasattr(lm, 'visibility') else True

# ==================== Pose Detection Functions ====================
def is_squat(lm) -> float:
    """ตรวจจับท่า Bodyweight Squat"""
    try:
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        R_knee = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        L_knee = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
        L_ankle = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        
        if not all([check_visibility(lm[i.value]) for i in [
            mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE
        ]]):
            return 0.0
    except:
        return 0.0
    
    R_knee_angle = angle_between(R_hip, R_knee, R_ankle)
    L_knee_angle = angle_between(L_hip, L_knee, L_ankle)
    avg_knee_angle = (R_knee_angle + L_knee_angle) / 2
    
    hip_y = (R_hip[1] + L_hip[1]) / 2
    knee_y = (R_knee[1] + L_knee[1]) / 2
    sh_y = (R_sh[1] + L_sh[1]) / 2
    
    # คะแนนจากมุมเข่า
    if 70 <= avg_knee_angle <= 110:
        angle_score = 1.0
    elif 110 < avg_knee_angle <= 140:
        angle_score = np.interp(avg_knee_angle, [110, 140], [1.0, 0.5])
    elif 60 <= avg_knee_angle < 70:
        angle_score = np.interp(avg_knee_angle, [60, 70], [0.7, 1.0])
    else:
        angle_score = np.interp(avg_knee_angle, [140, 180], [0.5, 0.0])
    
    # คะแนนจากความลึก
    depth_score = 1.0 if hip_y >= knee_y else np.interp(hip_y, [sh_y, knee_y], [0.0, 1.0])
    
    # คะแนนจากการทรงตัว
    hip_center_x = (R_hip[0] + L_hip[0]) / 2
    sh_center_x = (R_sh[0] + L_sh[0]) / 2
    balance_score = 1.0 - min(abs(hip_center_x - sh_center_x) * 3, 1.0)
    
    final_score = angle_score * 0.5 + depth_score * 0.35 + balance_score * 0.15
    return float(np.clip(final_score, 0.0, 1.0))

def is_pushup(lm) -> float:
    """ตรวจจับท่า Push-ups"""
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        L_el = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        L_wr = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        R_el = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        R_wr = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_ankle = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        
        if not check_visibility(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]):
            return 0.0
    except:
        return 0.0
    
    L_elbow_angle = angle_between(L_sh, L_el, L_wr)
    R_elbow_angle = angle_between(R_sh, R_el, R_wr)
    avg_elbow_angle = (L_elbow_angle + R_elbow_angle) / 2
    
    # คะแนนจากมุมศอก
    if 60 <= avg_elbow_angle <= 100:
        angle_score = 1.0
    elif 100 < avg_elbow_angle <= 140:
        angle_score = np.interp(avg_elbow_angle, [100, 140], [1.0, 0.5])
    else:
        angle_score = np.interp(avg_elbow_angle, [140, 180], [0.5, 0.0])
    
    # ความตรงของลำตัว
    shoulder_y = (L_sh[1] + R_sh[1]) / 2
    hip_y = (L_hip[1] + R_hip[1]) / 2
    ankle_y = (L_ankle[1] + R_ankle[1]) / 2
    
    sh_hip_diff = abs(shoulder_y - hip_y)
    hip_ankle_diff = abs(hip_y - ankle_y)
    
    torso_flatness = 1.0 - np.clip(sh_hip_diff * 8, 0, 1)
    leg_alignment = 1.0 - np.clip(hip_ankle_diff * 5, 0, 1)
    alignment_score = (torso_flatness + leg_alignment) / 2
    
    elbow_position_score = 1.0 if L_el[1] > L_sh[1] else 0.5
    
    final_score = angle_score * 0.5 + alignment_score * 0.4 + elbow_position_score * 0.1
    return float(np.clip(final_score, 0.0, 1.0))

def is_plank(lm) -> float:
    """ตรวจจับท่า Plank"""
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_ankle = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_el = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        R_el = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    except:
        return 0.0
    
    shoulder_y = (L_sh[1] + R_sh[1]) / 2
    hip_y = (L_hip[1] + R_hip[1]) / 2
    ankle_y = (L_ankle[1] + R_ankle[1]) / 2
    elbow_y = (L_el[1] + R_el[1]) / 2
    
    total_body_length = abs(shoulder_y - ankle_y)
    if total_body_length < 1e-6:
        return 0.0
    
    sh_hip_diff = abs(shoulder_y - hip_y)
    hip_ankle_diff = abs(hip_y - ankle_y)
    
    torso_flat = 1.0 - min(sh_hip_diff / (total_body_length * 0.5 + 1e-6), 1.0)
    legs_flat = 1.0 - min(hip_ankle_diff / (total_body_length * 0.5 + 1e-6), 1.0)
    alignment_score = (torso_flat + legs_flat) / 2
    
    hip_level_diff = hip_y - shoulder_y
    if -0.05 <= hip_level_diff <= 0.08:
        hip_score = 1.0
    elif hip_level_diff > 0.08:
        hip_score = np.interp(hip_level_diff, [0.08, 0.2], [1.0, 0.0])
    else:
        hip_score = np.interp(hip_level_diff, [-0.15, -0.05], [0.0, 1.0])
    
    elbow_position_score = 1.0 if elbow_y > shoulder_y else 0.6
    
    final_score = alignment_score * 0.5 + hip_score * 0.35 + elbow_position_score * 0.15
    return float(np.clip(final_score, 0.0, 1.0))

def is_situp(lm) -> float:
    """ตรวจจับท่า Sit-ups"""
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        nose = landmark_xy(lm[mp_pose.PoseLandmark.NOSE.value])
    except:
        return 0.0
    
    shoulder_y = (L_sh[1] + R_sh[1]) / 2
    hip_y = (L_hip[1] + R_hip[1]) / 2
    dist = abs(shoulder_y - hip_y)
    
    if dist <= 0.08:
        distance_score = 1.0
    elif dist <= 0.18:
        distance_score = np.interp(dist, [0.08, 0.18], [1.0, 0.5])
    else:
        distance_score = np.interp(dist, [0.18, 0.35], [0.5, 0.0])
    
    forward_score = 1.0 if nose[1] < hip_y + 0.1 else 0.7
    
    final_score = distance_score * 0.8 + forward_score * 0.2
    return float(np.clip(final_score, 0.0, 1.0))

def is_forward_lunge(lm) -> float:
    """ตรวจจับท่า Forward Lunge"""
    try:
        R_knee = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_knee = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        L_ankle = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    except:
        return 0.0
    
    r_forward = R_knee[0] - R_hip[0]
    l_forward = L_knee[0] - L_hip[0]
    forward_dist = max(abs(r_forward), abs(l_forward))
    
    R_angle = angle_between(R_hip, R_knee, R_ankle)
    L_angle = angle_between(L_hip, L_knee, L_ankle)
    front_angle = min(R_angle, L_angle)
    
    distance_score = np.clip(np.interp(forward_dist, [0.05, 0.25], [0, 1]), 0, 1)
    angle_score = 1.0 if 80 <= front_angle <= 110 else np.interp(front_angle, [60, 140], [0.5, 0.5])
    
    final_score = distance_score * 0.6 + angle_score * 0.4
    return float(np.clip(final_score, 0.0, 1.0))

def is_dead_bug(lm) -> float:
    """ตรวจจับท่า Dead Bug"""
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_wr = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
        R_wr = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        L_an = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_an = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    except:
        return 0.0
    
    shoulder_y = (L_sh[1] + R_sh[1]) / 2
    hip_y = (L_hip[1] + R_hip[1]) / 2
    
    torso_flat = 1.0 - np.clip(abs(shoulder_y - hip_y) * 10, 0, 1)
    
    limb_raise_scores = [
        np.interp(hip_y - L_wr[1], [0, 0.35], [0, 1]),
        np.interp(hip_y - R_wr[1], [0, 0.35], [0, 1]),
        np.interp(hip_y - L_an[1], [0, 0.35], [0, 1]),
        np.interp(hip_y - R_an[1], [0, 0.35], [0, 1])
    ]
    limb_raise = max(limb_raise_scores)
    
    final_score = torso_flat * 0.5 + limb_raise * 0.5
    return float(np.clip(final_score, 0.0, 1.0))

def is_side_plank(lm) -> float:
    """ตรวจจับท่า Side Plank"""
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_ankle = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    except:
        return 0.0
    
    hip_diff = abs(L_hip[1] - R_hip[1])
    L_alignment = angle_between(L_sh, L_hip, L_ankle)
    R_alignment = angle_between(R_sh, R_hip, R_ankle)
    alignment = max(L_alignment, R_alignment)
    
    side_score = np.clip(np.interp(hip_diff, [0.03, 0.2], [0, 1]), 0, 1)
    straightness = 1.0 if alignment > 160 else np.interp(alignment, [140, 160], [0.5, 1.0])
    
    final_score = side_score * 0.6 + straightness * 0.4
    return float(np.clip(final_score, 0.0, 1.0))

def is_russian_twist(lm) -> float:
    """ตรวจจับท่า Russian Twist"""
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    except:
        return 0.0
    
    shoulder_center_x = (L_sh[0] + R_sh[0]) / 2
    hip_center_x = (L_hip[0] + R_hip[0]) / 2
    twist = abs(shoulder_center_x - hip_center_x)
    
    sitting_score = 1.0 if (L_hip[1] + R_hip[1]) / 2 > (L_sh[1] + R_sh[1]) / 2 else 0.5
    
    twist_score = np.clip(np.interp(twist, [0.03, 0.15], [0, 1]), 0, 1)
    final_score = twist_score * 0.7 + sitting_score * 0.3
    return float(np.clip(final_score, 0.0, 1.0))

def is_lying_leg_raises(lm) -> float:
    """ตรวจจับท่า Lying Leg Raises"""
    try:
        L_an = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_an = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    except:
        return 0.0
    
    avg_an_y = (L_an[1] + R_an[1]) / 2
    avg_hip_y = (L_hip[1] + R_hip[1]) / 2
    avg_sh_y = (L_sh[1] + R_sh[1]) / 2
    
    leg_raise = avg_hip_y - avg_an_y
    torso_flat = 1.0 - np.clip(abs(avg_sh_y - avg_hip_y) * 8, 0, 1)
    
    raise_score = np.clip(np.interp(leg_raise, [0.05, 0.4], [0, 1]), 0, 1)
    final_score = raise_score * 0.7 + torso_flat * 0.3
    return float(np.clip(final_score, 0.0, 1.0))

# ==================== Detector Mapping ====================
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
REPS_POSES = {
    "Bodyweight Squat", "Push-ups", "Sit-ups", "Lunge (Forward Lunge)",
    "Dead Bug", "Russian Twist", "Lying Leg Raises"
}

# ==================== Feedback Functions ====================
def feedback_squat(lm, confidence: float) -> str:
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
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    R_angle = angle_between(R_hip, R_knee, R_ankle)
    L_angle = angle_between(L_hip, L_knee, L_ankle)
    avg_angle = (R_angle + L_angle) / 2
    
    hip_y = (R_hip[1] + L_hip[1]) / 2
    knee_y = (R_knee[1] + L_knee[1]) / 2
    sh_y = (R_sh[1] + L_sh[1]) / 2
    
    if confidence < 0.5:
        return "ยืนให้ตรง เตรียมพร้อมทำท่า Squat"
    elif avg_angle > 150:
        return "งอเข่าลงให้มากขึ้น ค่อยๆ นั่งลง"
    elif avg_angle < 60:
        return "ระวัง! อย่าง่อเข่าลึกเกินไป อาจบาดเจ็บ"
    elif hip_y < knee_y - 0.05:
        return "นั่งให้ลึกกว่านี้ สะโพกควรต่ำกว่าเข่า"
    elif hip_y > sh_y + 0.15:
        return "ลุกขึ้นแล้ว เริ่มรอบใหม่"
    else:
        return "ท่าถูกต้อง! ยอดเยี่ยม"

def feedback_pushup(lm, confidence: float) -> str:
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        L_el = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        L_wr = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    except:
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    elbow_angle = angle_between(L_sh, L_el, L_wr)
    shoulder_y = (L_sh[1] + R_sh[1]) / 2
    hip_y = (L_hip[1] + R_hip[1]) / 2
    torso_diff = abs(shoulder_y - hip_y)
    
    if confidence < 0.5:
        return "เข้าท่า Plank เตรียมทำ Push-up"
    elif torso_diff > 0.12:
        return "เก็บสะโพก! อย่าให้โก่งหลังหรือยกก้นสูง"
    elif elbow_angle > 160:
        return "งอศอกลงให้ลึกขึ้น"
    elif elbow_angle < 50:
        return "ดีแล้ว! ดันตัวขึ้นเลย"
    else:
        return "ท่าถูกต้อง! สุดยอด"

def feedback_plank(lm, confidence: float, hold_time: float) -> str:
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    except:
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    hip_y = (L_hip[1] + R_hip[1]) / 2
    hip_diff = hip_y - L_sh[1]
    
    if confidence < 0.5:
        return "เข้าท่า Plank ศอกแนบพื้น"
    elif hip_diff > 0.12:
        return "เก็บสะโพกลง อย่ายกสูงเกินไป"
    elif hip_diff < -0.08:
        return "ยกสะโพกขึ้นหน่อย อย่าให้ห้อยต่ำ"
    elif hold_time > 0:
        return f"กำลังดี! ทำไปแล้ว {hold_time:.1f}s"
    else:
        return "ท่าถูกต้อง! พยายามหายใจสม่ำเสมอ"

def feedback_situp(lm, confidence: float) -> str:
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    except:
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    dist = abs(L_sh[1] - L_hip[1])
    
    if confidence < 0.4:
        return "นอนราบ เตรียมทำ Sit-up"
    elif dist > 0.2:
        return "งอตัวขึ้นมาให้ใกล้เข่ามากขึ้น"
    elif dist < 0.08:
        return "ดีมาก! ลงอีกรอบ"
    else:
        return "ท่าถูกต้อง! เยี่ยมมาก"

def feedback_forward_lunge(lm, confidence: float) -> str:
    try:
        R_knee = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
        R_ankle = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_knee = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    except:
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    R_angle = angle_between(R_hip, R_knee, R_ankle)
    L_angle = angle_between(L_hip, L_knee, L_ankle)
    min_angle = min(R_angle, L_angle)
    
    r_forward = abs(R_knee[0] - R_hip[0])
    l_forward = abs(L_knee[0] - L_hip[0])
    
    if confidence < 0.4:
        return "ยืนตรง เตรียมก้าวขาออกไป"
    elif max(r_forward, l_forward) < 0.08:
        return "ก้าวขาออกไปให้ไกลขึ้น"
    elif min_angle > 130:
        return "งอเข่าหน้าลงให้มากขึ้น"
    else:
        return "ท่าถูกต้อง! เปลี่ยนขา"

def feedback_dead_bug(lm, confidence: float) -> str:
    try:
        L_wr = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
        R_wr = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        L_an = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_an = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
    except:
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    avg_arm_y = (L_wr[1] + R_wr[1]) / 2
    avg_leg_y = (L_an[1] + R_an[1]) / 2
    hip_y = L_hip[1]
    
    if confidence < 0.4:
        return "นอนหงาย เตรียมยกแขนและขา"
    elif avg_arm_y > hip_y or avg_leg_y > hip_y:
        return "ยกแขนและขาให้สูงขึ้น"
    else:
        return "ท่าถูกต้อง! สลับข้าง"

def feedback_side_plank(lm, confidence: float, hold_time: float) -> str:
    try:
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    except:
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    hip_diff = abs(L_hip[1] - R_hip[1])
    
    if confidence < 0.5:
        return "หันตัวข้าง เข้าท่า Side Plank"
    elif hip_diff < 0.05:
        return "ยกตัวขึ้น แยกสะโพกออกจากพื้น"
    elif hold_time > 0:
        return f"ดีมาก! ทำไปแล้ว {hold_time:.1f}s"
    else:
        return "ท่าถูกต้อง! เก็บสะโพกให้สูง"

def feedback_russian_twist(lm, confidence: float) -> str:
    try:
        L_sh = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        R_sh = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    except:
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    sh_center_x = (L_sh[0] + R_sh[0]) / 2
    hip_center_x = (L_hip[0] + R_hip[0]) / 2
    twist = abs(sh_center_x - hip_center_x)
    
    if confidence < 0.4:
        return "นั่ง เตรียมบิดลำตัว"
    elif twist < 0.05:
        return "บิดลำตัวไปซ้ายและขวามากขึ้น"
    else:
        return "ท่าถูกต้อง! บิดต่อไป"

def feedback_lying_leg_raises(lm, confidence: float) -> str:
    try:
        L_an = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        R_an = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        L_hip = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        R_hip = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
    except:
        return "ไม่สามารถตรวจจับท่าได้ชัดเจน"
    
    avg_an_y = (L_an[1] + R_an[1]) / 2
    avg_hip_y = (L_hip[1] + R_hip[1]) / 2
    
    if confidence < 0.4:
        return "นอนราบ เตรียมยกขาขึ้น"
    elif avg_an_y > avg_hip_y + 0.05:
        return "ยกขาขึ้นให้สูงกว่านี้"
    elif avg_an_y < avg_hip_y - 0.35:
        return "ดีมาก! ลดขาลง"
    else:
        return "ท่าถูกต้อง! ยกต่อไป"

# ==================== Feedback Mapping ====================
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

# ==================== Counter Update ====================
def update_counters(client_id: str, pose_name: str, confidence: float, ts: float) -> None:
    """อัพเดท counters ด้วยความแม่นยำสูง"""
    client_state = client_states[client_id]
    
    if pose_name in HOLD_POSES:
        # Hold-based exercises
        if pose_name not in client_state.hold_timers:
            client_state.hold_timers[pose_name] = HoldTimer()
        
        ht = client_state.hold_timers[pose_name]
        
        if confidence >= config.HOLD_MIN_CONFIDENCE:
            ht.current_streak += 1
            if ht.started_at is None and ht.current_streak >= config.MIN_CONFIDENCE_HISTORY:
                ht.started_at = ts
            elif ht.started_at is not None:
                hold_duration = ts - ht.started_at
                if hold_duration > ht.best:
                    ht.best = hold_duration
        else:
            if ht.started_at is not None:
                hold_duration = ts - ht.started_at
                if hold_duration > ht.best:
                    ht.best = hold_duration
                ht.started_at = None
            ht.current_streak = 0
    else:
        # Rep-based exercises
        if pose_name not in client_state.pose_states:
            client_state.pose_states[pose_name] = PoseState()
        
        state = client_state.pose_states[pose_name]
        state.confidence_history.append(confidence)
        
        avg_confidence = sum(state.confidence_history) / len(state.confidence_history)
        
        if avg_confidence >= config.POSE_START_THRESH and not state.in_pose:
            if len(state.confidence_history) >= config.MIN_CONFIDENCE_HISTORY:
                state.in_pose = True
                state.last_detection_time = ts
        
        elif avg_confidence <= config.POSE_END_THRESH and state.in_pose:
            if ts - state.last_detection_time >= config.DETECTION_COOLDOWN:
                if pose_name not in client_state.reps_counts:
                    client_state.reps_counts[pose_name] = 0
                client_state.reps_counts[pose_name] += 1
                state.in_pose = False
                state.confidence_history.clear()

# ==================== WebSocket Endpoint ====================
@app.websocket("/ws/pose")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}_{int(time.time()*1000)}"
    logger.info(f"[CONNECTED] {client_id}")
    
    client_states[client_id] = ClientState()
    frame_idx = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=config.MODEL_COMPLEXITY,
        enable_segmentation=False,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        smooth_landmarks=True
    ) as pose_detector:
        try:
            while True:
                data = await websocket.receive_text()
                frame_idx += 1
                
                # Handle Commands
                if data.startswith("{"):
                    try:
                        cmd = json.loads(data)
                        
                        if "frame_skip" in cmd:
                            client_states[client_id].frame_skip = int(cmd["frame_skip"])
                        
                        if "select_pose" in cmd:
                            pose_name = cmd["select_pose"]
                            if pose_name in DETECTORS:
                                client_states[client_id].selected_pose = pose_name
                                logger.info(f"[{client_id}] Selected pose: {pose_name}")
                                await websocket.send_text(json.dumps({
                                    "status": "pose_selected",
                                    "pose": pose_name
                                }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "status": "error",
                                    "detail": f"Unknown pose: {pose_name}"
                                }))
                    except Exception as e:
                        logger.error(f"[CMD ERROR] {e}")
                    continue
                
                # Frame Skip
                if (frame_idx % (client_states[client_id].frame_skip + 1)) != 0:
                    await websocket.send_text(json.dumps({"status": "ok"}))
                    continue
                
                # Decode Frame
                try:
                    frame = cv2.imdecode(
                        np.frombuffer(base64.b64decode(data), np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        raise ValueError("Frame decode failed")
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "error": "decode_failed",
                        "detail": str(e)
                    }))
                    continue
                
                ts = time.time()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_detector.process(rgb_frame)
                
                # Prepare Response
                response = {
                    "pose": "N/A",
                    "confidence": 0.0,
                    "reps": client_states[client_id].reps_counts.copy(),
                    "holds": {},
                    "timestamp": ts,
                    "selected_pose": client_states[client_id].selected_pose,
                    "advice": ""
                }
                
                selected_pose = client_states[client_id].selected_pose
                
                # No Landmarks Detected
                if not results.pose_landmarks:
                    response["advice"] = "กรุณาเข้ามาในกรอบกล้อง" if selected_pose else "ยังไม่ได้เลือกท่าออกกำลังกาย"
                    await websocket.send_text(json.dumps(response))
                    continue
                
                landmarks = results.pose_landmarks.landmark
                
                # Process Selected Pose
                if selected_pose and selected_pose in DETECTORS:
                    try:
                        confidence = DETECTORS[selected_pose](landmarks)
                        update_counters(client_id, selected_pose, confidence, ts)
                        
                        # Collect holds data
                        user_holds = {}
                        for p, h in client_states[client_id].hold_timers.items():
                            current_hold = round((ts - h.started_at) if h.started_at else 0.0, 2)
                            user_holds[p] = {
                                "current_hold": current_hold,
                                "best_hold": round(h.best, 2)
                            }
                        
                        # Generate feedback
                        advice = ""
                        if ts - client_states[client_id].last_feedback_time >= config.FEEDBACK_INTERVAL:
                            try:
                                if selected_pose in FEEDBACKS:
                                    if selected_pose in HOLD_POSES:
                                        current_hold = user_holds.get(selected_pose, {}).get("current_hold", 0)
                                        advice = FEEDBACKS[selected_pose](landmarks, confidence, current_hold)
                                    else:
                                        advice = FEEDBACKS[selected_pose](landmarks, confidence)
                                    
                                    # เก็บ advice เพื่อส่งต่อเนื่อง
                                    if advice:
                                        client_states[client_id].last_advice = advice
                                        client_states[client_id].last_feedback_time = ts
                            except Exception as e:
                                logger.error(f"[FEEDBACK ERROR] {e}")
                                advice = "กำลังวิเคราะห์ท่า..."
                        else:
                            # ใช้ advice ล่าสุด
                            advice = client_states[client_id].last_advice
                        
                        response.update({
                            "pose": selected_pose,
                            "confidence": round(confidence, 3),
                            "reps": client_states[client_id].reps_counts.copy(),
                            "holds": user_holds,
                            "advice": advice
                        })
                    
                    except Exception as e:
                        logger.error(f"[PROCESSING ERROR] {e}")
                        response["advice"] = "เกิดข้อผิดพลาดในการวิเคราะห์"
                else:
                    response["advice"] = "กรุณาเลือกท่าที่ต้องการออกกำลังกาย"
                
                await websocket.send_text(json.dumps(response))
        
        except WebSocketDisconnect:
            logger.info(f"[DISCONNECTED] {client_id}")
        except Exception as e:
            logger.error(f"[UNEXPECTED ERROR] {e}")
        finally:
            if client_id in client_states:
                del client_states[client_id]

# ==================== HTTP Endpoints ====================
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "status": "online",
        "service": "Pose Detection API",
        "version": "2.0.0",
        "available_poses": list(DETECTORS.keys()),
        "active_clients": len(client_states),
        "endpoints": {
            "websocket": "/ws/pose",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_clients": len(client_states),
        "timestamp": time.time()
    }

@app.get("/stats")
async def stats():
    """Statistics endpoint"""
    total_reps = sum(
        sum(state.reps_counts.values())
        for state in client_states.values()
    )
    
    total_hold_time = sum(
        sum(timer.best for timer in state.hold_timers.values())
        for state in client_states.values()
    )
    
    return {
        "active_clients": len(client_states),
        "total_reps": total_reps,
        "total_hold_time": round(total_hold_time, 2),
        "poses_in_use": list(set(
            state.selected_pose
            for state in client_states.values()
            if state.selected_pose
        )),
        "timestamp": time.time()
    }

@app.get("/poses")
async def list_poses():
    """List all available poses"""
    return {
        "hold_poses": list(HOLD_POSES),
        "rep_poses": list(REPS_POSES),
        "all_poses": list(DETECTORS.keys())
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Pose Detection API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

# Run with: python main.py
# Or: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000