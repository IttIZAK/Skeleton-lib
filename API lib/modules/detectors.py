# modules/detectors.py - IMPROVED VERSION
import numpy as np
from .utils import landmark_xy, angle, check_visibility

def _get(lm, mp, name):
    """Helper: return lm[index] where index is mp.PoseLandmark.<name>.value"""
    return lm[getattr(mp.PoseLandmark, name).value]

def _check_landmarks_visible(lm, mp, landmark_names, min_visibility=0.5):
    """ตรวจสอบว่า landmarks ที่ระบุมองเห็นได้หรือไม่"""
    visible_count = 0
    for name in landmark_names:
        try:
            landmark = _get(lm, mp, name)
            if check_visibility(landmark, min_visibility):
                visible_count += 1
        except Exception:
            pass
    
    required_ratio = 0.7
    return visible_count >= len(landmark_names) * required_ratio

def detect_squat(lm, mp):
    """Squat Detection"""
    required = ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", 
                "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
                "RIGHT_SHOULDER", "LEFT_SHOULDER"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.4):
        return 0.0
    
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

    R_knee_angle = angle(R_hip, R_knee, R_ankle)
    L_knee_angle = angle(L_hip, L_knee, L_ankle)
    avg_knee_angle = (R_knee_angle + L_knee_angle) / 2
    
    if 70 <= avg_knee_angle <= 100:
        knee_score = 1.0
    elif 100 < avg_knee_angle <= 140:
        knee_score = np.interp(avg_knee_angle, [100, 140], [0.8, 0.3])
    elif avg_knee_angle > 140:
        knee_score = 0.1
    else:
        knee_score = np.interp(avg_knee_angle, [50, 70], [0.5, 1.0])
    
    R_torso_angle = angle(R_sh, R_hip, R_knee)
    L_torso_angle = angle(L_sh, L_hip, L_knee)
    avg_torso_angle = (R_torso_angle + L_torso_angle) / 2
    
    if 30 <= avg_torso_angle <= 90:
        torso_score = 1.0
    elif avg_torso_angle < 30:
        torso_score = np.interp(avg_torso_angle, [10, 30], [0.3, 1.0])
    else:
        torso_score = np.interp(avg_torso_angle, [90, 120], [1.0, 0.4])
    
    hip_center_x = (R_hip[0] + L_hip[0]) / 2
    sh_center_x = (R_sh[0] + L_sh[0]) / 2
    balance_score = 1.0 - min(abs(hip_center_x - sh_center_x) * 2.0, 0.3)
    
    final_score = (knee_score * 0.5 + torso_score * 0.4 + balance_score * 0.1)
    
    return float(np.clip(final_score, 0, 1))

def detect_pushup(lm, mp):
    """Push-up Detection"""
    required = ["RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_WRIST",
                "LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_WRIST",
                "RIGHT_HIP", "LEFT_HIP", "RIGHT_ANKLE", "LEFT_ANKLE"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.4):
        return 0.0
    
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
    avg_elbow = (R_angle + L_angle) / 2
    
    if avg_elbow < 100:
        elbow_score = 1.0
    elif avg_elbow < 140:
        elbow_score = np.interp(avg_elbow, [100, 140], [1.0, 0.5])
    else:
        elbow_score = np.interp(avg_elbow, [140, 170], [0.5, 0.2])
    
    hip_y = (R_hip[1] + L_hip[1]) / 2
    ankle_y = (R_ankle[1] + L_ankle[1]) / 2
    sh_y = (R_sh[1] + L_sh[1]) / 2
    
    torso_len = abs(sh_y - hip_y) + 1e-6
    expected_hip_y = (sh_y + ankle_y) / 2
    deviation = abs(hip_y - expected_hip_y)
    
    straight_score = 1.0 - min(deviation / torso_len * 1.0, 0.3)
    
    body_angle = angle(
        ((R_sh[0] + L_sh[0])/2, (R_sh[1] + L_sh[1])/2),
        ((R_hip[0] + L_hip[0])/2, (R_hip[1] + L_hip[1])/2),
        ((R_ankle[0] + L_ankle[0])/2, (R_ankle[1] + L_ankle[1])/2)
    )
    alignment_score = np.interp(body_angle, [140, 180], [0.5, 1.0])
    alignment_score = np.clip(alignment_score, 0, 1)
    
    final_score = (elbow_score * 0.7 + straight_score * 0.15 + alignment_score * 0.15)
    
    return float(np.clip(final_score, 0, 1))

def detect_plank(lm, mp):
    """Plank Detection"""
    required = ["RIGHT_SHOULDER", "LEFT_SHOULDER", 
                "RIGHT_HIP", "LEFT_HIP",
                "RIGHT_ANKLE", "LEFT_ANKLE"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.7):
        return 0.0
    
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
    expected_hip = (sh_y + ankle_y) / 2
    deviation = abs(hip_y - expected_hip)
    
    straight_score = 1.0 - min(deviation / torso_len * 1.5, 1.0)
    
    hip_drop = hip_y - expected_hip
    if abs(hip_drop) < torso_len * 0.1:
        position_score = 1.0
    else:
        position_score = max(0.5, 1.0 - abs(hip_drop) / torso_len)
    
    final_score = (straight_score * 0.7 + position_score * 0.3)
    
    return float(np.clip(final_score, 0, 1))

def detect_situp(lm, mp):
    """Sit-up Detection"""
    required = ["RIGHT_SHOULDER", "LEFT_SHOULDER", 
                "RIGHT_HIP", "LEFT_HIP",
                "RIGHT_KNEE", "LEFT_KNEE"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.4):
        return 0.0
    
    try:
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_knee = landmark_xy(_get(lm, mp, "RIGHT_KNEE"))
        L_knee = landmark_xy(_get(lm, mp, "LEFT_KNEE"))
    except Exception:
        return 0.0
    
    R_torso_angle = angle(R_sh, R_hip, R_knee)
    L_torso_angle = angle(L_sh, L_hip, L_knee)
    avg_angle = (R_torso_angle + L_torso_angle) / 2
    
    if 40 <= avg_angle <= 70:
        score = 1.0
    elif 70 < avg_angle <= 100:
        score = np.interp(avg_angle, [70, 100], [0.8, 0.3])
    elif avg_angle < 40:
        score = np.interp(avg_angle, [20, 40], [0.5, 1.0])
    else:
        score = np.interp(avg_angle, [100, 130], [0.3, 0.0])
    
    return float(np.clip(score, 0, 1))

def detect_lunge(lm, mp):
    """Lunge Detection"""
    required = ["RIGHT_KNEE", "LEFT_KNEE",
                "RIGHT_HIP", "LEFT_HIP",
                "RIGHT_ANKLE", "LEFT_ANKLE",
                "RIGHT_SHOULDER", "LEFT_SHOULDER"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.3):
        return 0.0
    
    try:
        R_knee = landmark_xy(_get(lm, mp, "RIGHT_KNEE"))
        L_knee = landmark_xy(_get(lm, mp, "LEFT_KNEE"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
    except Exception:
        return 0.0
    
    R_knee_angle = angle(R_hip, R_knee, R_ankle)
    L_knee_angle = angle(L_hip, L_knee, L_ankle)
    
    if R_knee_angle < L_knee_angle:
        front_angle = R_knee_angle
        back_angle = L_knee_angle
        front_knee_y = R_knee[1]
        back_knee_y = L_knee[1]
    else:
        front_angle = L_knee_angle
        back_angle = R_knee_angle
        front_knee_y = L_knee[1]
        back_knee_y = R_knee[1]
    
    if 60 <= front_angle <= 110:
        front_score = 1.0
    elif 110 < front_angle <= 130:
        front_score = np.interp(front_angle, [110, 130], [1.0, 0.4])
    elif front_angle < 60:
        front_score = np.interp(front_angle, [40, 60], [0.3, 1.0])
    else:
        front_score = 0.3
    
    if back_angle > 120:
        back_score = 1.0
    elif 100 < back_angle <= 120:
        back_score = np.interp(back_angle, [100, 120], [0.5, 1.0])
    else:
        back_score = 0.4
    
    angle_diff = abs(front_angle - back_angle)
    if angle_diff > 30:
        diff_score = 1.0
    elif angle_diff > 20:
        diff_score = np.interp(angle_diff, [20, 30], [0.5, 1.0])
    else:
        diff_score = 0.5
    
    knee_height_diff = back_knee_y - front_knee_y
    if knee_height_diff > 0.03:
        position_score = 1.0
    elif knee_height_diff > 0.0:
        position_score = np.interp(knee_height_diff, [0.0, 0.03], [0.6, 1.0])
    else:
        position_score = 0.6
    
    sh_center_x = (R_sh[0] + L_sh[0]) / 2
    hip_center_x = (R_hip[0] + L_hip[0]) / 2
    lean = abs(sh_center_x - hip_center_x)
    
    if lean < 0.15:
        torso_score = 1.0
    else:
        torso_score = max(0.6, 1.0 - lean * 2)
    
    final_score = (front_score * 0.40 + back_score * 0.25 + 
                   diff_score * 0.15 + position_score * 0.10 + torso_score * 0.10)
    
    return float(np.clip(final_score, 0, 1))

def detect_dead_bug(lm, mp):
    """Dead Bug Detection"""
    required = ["LEFT_WRIST", "RIGHT_WRIST",
                "LEFT_ELBOW", "RIGHT_ELBOW",
                "LEFT_ANKLE", "RIGHT_ANKLE",
                "LEFT_KNEE", "RIGHT_KNEE",
                "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.3):
        return 0.0
    
    try:
        L_wrist = landmark_xy(_get(lm, mp, "LEFT_WRIST"))
        R_wrist = landmark_xy(_get(lm, mp, "RIGHT_WRIST"))
        L_elbow = landmark_xy(_get(lm, mp, "LEFT_ELBOW"))
        R_elbow = landmark_xy(_get(lm, mp, "RIGHT_ELBOW"))
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_knee = landmark_xy(_get(lm, mp, "LEFT_KNEE"))
        R_knee = landmark_xy(_get(lm, mp, "RIGHT_KNEE"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
    except Exception:
        return 0.0
    
    R_arm_angle = angle(R_sh, R_elbow, R_wrist)
    L_arm_angle = angle(L_sh, L_elbow, L_wrist)
    avg_arm_angle = (R_arm_angle + L_arm_angle) / 2
    
    if 110 <= avg_arm_angle <= 180:
        arm_score = 1.0
    elif avg_arm_angle < 110:
        arm_score = np.interp(avg_arm_angle, [80, 110], [0.4, 1.0])
    else:
        arm_score = 0.7
    
    R_leg_angle = angle(R_hip, R_knee, R_ankle)
    L_leg_angle = angle(L_hip, L_knee, L_ankle)
    avg_leg_angle = (R_leg_angle + L_leg_angle) / 2
    
    if 110 <= avg_leg_angle <= 180:
        leg_score = 1.0
    elif avg_leg_angle < 110:
        leg_score = np.interp(avg_leg_angle, [80, 110], [0.4, 1.0])
    else:
        leg_score = 0.7
    
    sh_y = (L_sh[1] + R_sh[1]) / 2
    hip_y = (L_hip[1] + R_hip[1]) / 2
    
    L_wrist_up = (sh_y - L_wrist[1]) > 0.02
    R_wrist_up = (sh_y - R_wrist[1]) > 0.02
    L_ankle_up = (hip_y - L_ankle[1]) > 0.02
    R_ankle_up = (hip_y - R_ankle[1]) > 0.02
    
    any_movement = L_wrist_up or R_wrist_up or L_ankle_up or R_ankle_up
    alternating = (L_wrist_up and R_ankle_up) or (R_wrist_up and L_ankle_up)
    
    if alternating:
        alternate_score = 1.0
    elif any_movement:
        alternate_score = 0.6
    else:
        alternate_score = 0.3
    
    final_score = (arm_score * 0.35 + leg_score * 0.35 + alternate_score * 0.30)
    
    return float(np.clip(final_score, 0, 1))

def detect_side_plank(lm, mp):
    """Side Plank Detection"""
    required = ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_ANKLE"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.7):
        return 0.0
    
    try:
        sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
    except Exception:
        return 0.0
    
    body_angle = angle(sh, hip, ankle)
    
    angle_score = np.interp(body_angle, [155, 180], [0.6, 1.0])
    angle_score = np.clip(angle_score, 0, 1)
    
    expected_hip_y = (sh[1] + ankle[1]) / 2
    hip_drop = hip[1] - expected_hip_y
    
    if hip_drop < 0.02:
        position_score = 1.0
    else:
        position_score = max(0.5, 1.0 - hip_drop * 5)
    
    final_score = (angle_score * 0.7 + position_score * 0.3)
    
    return float(np.clip(final_score, 0, 1))

def detect_russian_twist(lm, mp):
    """
    ✅ IMPROVED Russian Twist Detection
    - นับ Reps เมื่อบิดตัวไปมา (ซ้าย-ขวา)
    - ตรวจจับการหมุนไหล่ชัดเจน
    - Reset ง่ายเมื่อกลับกลางตัว
    """
    required = ["LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.4):
        return 0.0
    
    try:
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
    except Exception:
        return 0.0
    
    # 1. ✅ การหมุนไหล่ (ซ้าย-ขวา) - ตัวชี้วัดหลัก
    shoulder_x_diff = abs(L_sh[0] - R_sh[0])
    rotation_score = min(shoulder_x_diff * 6, 1.0)
    
    # 2. ✅ การเอียงลำตัว (twist)
    sh_center_x = (L_sh[0] + R_sh[0]) / 2
    hip_center_x = (L_hip[0] + R_hip[0]) / 2
    twist = abs(sh_center_x - hip_center_x)
    twist_score = min(twist * 5, 1.0)
    
    # 3. ✅ การโน้มตัวไปหลัง
    sh_y = (L_sh[1] + R_sh[1]) / 2
    hip_y = (L_hip[1] + R_hip[1]) / 2
    lean_back = sh_y - hip_y
    
    if lean_back > -0.02:
        lean_score = min(abs(lean_back) * 4, 1.0)
    else:
        lean_score = 0.5
    
    # ✅ CRITICAL: ตรวจจับว่าอยู่กลางตัว (ไม่หมุน) = confidence ต่ำ = reset ได้
    is_centered = (shoulder_x_diff < 0.05 and twist < 0.02)
    
    if is_centered:
        # กลางตัว = ให้ confidence ต่ำมาก เพื่อ reset
        final_score = 0.05
    else:
        # กำลังบิด = ให้ confidence สูง
        final_score = (rotation_score * 0.50 + 
                       twist_score * 0.35 + 
                       lean_score * 0.15)
    
    return float(np.clip(final_score, 0, 1))

def detect_lying_leg_raises(lm, mp):
    """
    ✅ IMPROVED Lying Leg Raises Detection
    - นับ Reps เมื่อยกขาสูงสุดแล้ว (มุม 20-80°)
    - ให้ confidence สูงเฉพาะเมื่อขาอยู่ในตำแหน่งที่ถูกต้อง
    """
    required = ["LEFT_ANKLE", "RIGHT_ANKLE",
                "LEFT_HIP", "RIGHT_HIP",
                "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_KNEE", "RIGHT_KNEE"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.3):
        return 0.0
    
    try:
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_knee = landmark_xy(_get(lm, mp, "LEFT_KNEE"))
        R_knee = landmark_xy(_get(lm, mp, "RIGHT_KNEE"))
    except Exception:
        return 0.0
    
    # 1. ✅ มุมสะโพก - ใช้ range กว้างขึ้น
    R_hip_angle = angle(R_sh, R_hip, R_ankle)
    L_hip_angle = angle(L_sh, L_hip, L_ankle)
    avg_angle = (R_hip_angle + L_hip_angle) / 2
    
    # ✅ ให้ confidence สูงเมื่อยกขาสูง (20-80°)
    if 20 <= avg_angle <= 80:
        score = 1.0
    elif 80 < avg_angle <= 110:
        score = np.interp(avg_angle, [80, 110], [1.0, 0.3])
    elif avg_angle < 20:
        score = np.interp(avg_angle, [5, 20], [0.5, 1.0])
    else:
        # ✅ ขาลงต่ำ = confidence ต่ำมาก เพื่อให้ reset ได้
        score = np.interp(avg_angle, [110, 140], [0.3, 0.05])
    
    # 2. ความตรงของขา
    R_leg_straight = angle(R_hip, R_knee, R_ankle)
    L_leg_straight = angle(L_hip, L_knee, L_ankle)
    avg_leg_straight = (R_leg_straight + L_leg_straight) / 2
    
    if avg_leg_straight > 150:
        straight_score = 1.0
    else:
        straight_score = np.interp(avg_leg_straight, [130, 150], [0.6, 1.0])
    
    # 3. ความสมมาตร
    ankle_diff = abs(L_ankle[1] - R_ankle[1])
    symmetry_score = 1.0 - min(ankle_diff * 1.5, 0.2)
    
    # ✅ ให้น้ำหนักกับมุมสะโพกเป็นหลัก
    final_score = (score * 0.70 + 
                   straight_score * 0.15 + 
                   symmetry_score * 0.15)
    
    return float(np.clip(final_score, 0, 1))