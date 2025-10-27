# modules/detectors.py
import numpy as np
from .utils import landmark_xy, angle

def _get(lm, mp, name):
    """Helper: return lm[index] where index is mp.PoseLandmark.<name>.value"""
    return lm[getattr(mp.PoseLandmark, name).value]

def detect_squat(lm, mp):
    try:
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        R_knee = landmark_xy(_get(lm, mp, "RIGHT_KNEE"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        L_knee = landmark_xy(_get(lm, mp, "LEFT_KNEE"))
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
    except Exception:
        return 0.0

    R_angle = angle(R_hip, R_knee, R_ankle)
    L_angle = angle(L_hip, L_knee, L_ankle)
    avg_knee = (R_angle + L_angle) / 2
    angle_score = np.interp(avg_knee, [60, 120], [0.7, 1.0])
    hip_y = (R_hip[1] + L_hip[1]) / 2
    knee_y = (R_knee[1] + L_knee[1]) / 2
    sh_y = (R_sh[1] + L_sh[1]) / 2
    torso_len = abs(sh_y - hip_y) + 1e-6
    depth_score = np.clip((hip_y - knee_y) / torso_len, 0, 1)
    hip_center_x = (R_hip[0] + L_hip[0]) / 2
    sh_center_x = (R_sh[0] + L_sh[0]) / 2
    balance_score = 1.0 - min(abs(hip_center_x - sh_center_x) * 3, 1.0)
    return float(np.clip(angle_score * 0.5 + depth_score * 0.35 + balance_score * 0.15, 0, 1))

def detect_pushup(lm, mp):
    try:
        R_el = landmark_xy(_get(lm, mp, "RIGHT_ELBOW"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        R_wr = landmark_xy(_get(lm, mp, "RIGHT_WRIST"))
        L_el = landmark_xy(_get(lm, mp, "LEFT_ELBOW"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        L_wr = landmark_xy(_get(lm, mp, "LEFT_WRIST"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
    except Exception:
        return 0.0

    R_angle = angle(R_sh, R_el, R_wr)
    L_angle = angle(L_sh, L_el, L_wr)
    avg = (R_angle + L_angle) / 2
    elbow_score = np.interp(avg, [60, 120], [0.7, 1.0])
    hip_y = (R_hip[1] + L_hip[1]) / 2
    ankle_y = (R_ankle[1] + L_ankle[1]) / 2
    sh_y = (R_sh[1] + L_sh[1]) / 2
    torso_len = abs(sh_y - hip_y) + 1e-6
    straight_score = 1.0 - min(abs(hip_y - (sh_y + ankle_y) / 2) / torso_len, 0.5)
    return float(np.clip(elbow_score * 0.7 + straight_score * 0.3, 0, 1))

def detect_plank(lm, mp):
    try:
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
    except Exception:
        return 0.0

    sh_y = (R_sh[1] + L_sh[1]) / 2
    hip_y = (R_hip[1] + L_hip[1]) / 2
    ankle_y = (R_ankle[1] + L_ankle[1]) / 2
    torso_len = abs(sh_y - hip_y) + 1e-6
    deviation = abs((sh_y + ankle_y) / 2 - hip_y)
    straight_score = 1.0 - min(deviation / torso_len, 0.5)
    return float(np.clip(straight_score, 0, 1))

def detect_situp(lm, mp):
    try:
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
    except Exception:
        return 0.0
    sh_y = (R_sh[1] + L_sh[1]) / 2
    hip_y = (R_hip[1] + L_hip[1]) / 2
    torso_len = abs(sh_y - hip_y) + 1e-6
    depth = (hip_y - sh_y) / torso_len
    return float(np.clip(np.interp(depth, [0, 0.4], [0, 1]), 0, 1))

def detect_lunge(lm, mp):
    try:
        R_knee = landmark_xy(_get(lm, mp, "RIGHT_KNEE"))
        L_knee = landmark_xy(_get(lm, mp, "LEFT_KNEE"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
    except Exception:
        return 0.0
    R_angle = angle(R_hip, R_knee, R_ankle)
    L_angle = angle(L_hip, L_knee, L_ankle)
    return float(np.clip(np.interp(min(R_angle, L_angle), [60, 90], [0, 1]), 0, 1))

def detect_dead_bug(lm, mp):
    try:
        L_hand = landmark_xy(_get(lm, mp, "LEFT_WRIST"))
        R_hand = landmark_xy(_get(lm, mp, "RIGHT_WRIST"))
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
    except Exception:
        return 0.0
    sh_y = (L_sh[1] + R_sh[1]) / 2
    hand_y = (L_hand[1] + R_hand[1]) / 2
    ankle_y = (L_ankle[1] + R_ankle[1]) / 2
    hand_score = 1.0 - min(abs(sh_y - hand_y) * 3, 1.0)
    leg_score = 1.0 - min(abs(sh_y - ankle_y) * 3, 1.0)
    return float(np.clip((hand_score + leg_score) / 2, 0, 1))

def detect_side_plank(lm, mp):
    try:
        sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
    except Exception:
        return 0.0
    ang = angle(sh, hip, ankle)
    return float(np.clip(np.interp(ang, [160, 180], [0, 1]), 0, 1))

def detect_russian_twist(lm, mp):
    try:
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
    except Exception:
        return 0.0
    sh_center_x = (L_sh[0] + R_sh[0]) / 2
    hip_center_x = (L_hip[0] + R_hip[0]) / 2
    twist = abs(sh_center_x - hip_center_x)
    hip_y = (L_hip[1] + R_hip[1]) / 2
    sh_y = (L_sh[1] + R_sh[1]) / 2
    twist_score = min(twist * 5, 1.0)
    lean_score = 1.0 - min((sh_y - hip_y) * 5, 1.0)
    return float(np.clip((twist_score * 0.6 + lean_score * 0.4), 0, 1))

def detect_lying_leg_raises(lm, mp):
    try:
        L_an = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
        R_an = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
    except Exception:
        return 0.0
    avg_an_y = (L_an[1] + R_an[1]) / 2
    avg_hip_y = (L_hip[1] + R_hip[1]) / 2
    leg_raise = avg_hip_y - avg_an_y
    return float(np.clip(np.interp(leg_raise, [0, 0.4], [0, 1]), 0, 1))
