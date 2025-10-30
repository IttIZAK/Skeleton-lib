# modules/detectors.py
import numpy as np
from .utils import landmark_xy, angle, check_visibility

def _get(lm, mp, name):
    """Helper: return lm[index] where index is mp.PoseLandmark.<name>.value"""
    return lm[getattr(mp.PoseLandmark, name).value]

def _check_landmarks_visible(lm, mp, landmark_names, min_visibility=0.5):
    """ตรวจสอบว่า landmarks ที่ระบุมองเห็นได้หรือไม่ - ผ่อนปรนขึ้น"""
    visible_count = 0
    for name in landmark_names:
        try:
            landmark = _get(lm, mp, name)
            if check_visibility(landmark, min_visibility):
                visible_count += 1
        except Exception:
            pass
    
    # ต้องเห็นอย่างน้อย 70% ของจุดที่ต้องการ (เดิมต้อง 100%)
    required_ratio = 0.7
    return visible_count >= len(landmark_names) * required_ratio

def detect_squat(lm, mp):
    """
    Squat Detection ตามเกณฑ์:
    - เข่า (HIP-KNEE-ANKLE): 70°-140°
    - ลำตัว (SHOULDER-HIP-KNEE): 30°-90°
    """
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

    # 1. มุมเข่า (HIP-KNEE-ANKLE): 70°-140°
    R_knee_angle = angle(R_hip, R_knee, R_ankle)
    L_knee_angle = angle(L_hip, L_knee, L_ankle)
    avg_knee_angle = (R_knee_angle + L_knee_angle) / 2
    
    # ยิ่งใกล้ 70° (ย่อลึก) = คะแนนสูง
    if 70 <= avg_knee_angle <= 100:
        knee_score = 1.0
    elif 100 < avg_knee_angle <= 140:
        knee_score = np.interp(avg_knee_angle, [100, 140], [0.8, 0.3])
    elif avg_knee_angle > 140:
        knee_score = 0.1  # ยืนตรงเกินไป
    else:
        knee_score = np.interp(avg_knee_angle, [50, 70], [0.5, 1.0])
    
    # 2. มุมลำตัว (SHOULDER-HIP-KNEE): 30°-90°
    R_torso_angle = angle(R_sh, R_hip, R_knee)
    L_torso_angle = angle(L_sh, L_hip, L_knee)
    avg_torso_angle = (R_torso_angle + L_torso_angle) / 2
    
    if 30 <= avg_torso_angle <= 90:
        torso_score = 1.0
    elif avg_torso_angle < 30:
        torso_score = np.interp(avg_torso_angle, [10, 30], [0.3, 1.0])
    else:
        torso_score = np.interp(avg_torso_angle, [90, 120], [1.0, 0.4])
    
    # 3. ความสมดุล
    hip_center_x = (R_hip[0] + L_hip[0]) / 2
    sh_center_x = (R_sh[0] + L_sh[0]) / 2
    balance_score = 1.0 - min(abs(hip_center_x - sh_center_x) * 2.0, 0.3)
    
    # คะแนนรวม
    final_score = (knee_score * 0.5 + torso_score * 0.4 + balance_score * 0.1)
    
    return float(np.clip(final_score, 0, 1))

def detect_pushup(lm, mp):
    """
    ปรับปรุง Push-up detection:
    - ตรวจสอบมุมศอก
    - ตรวจสอบความตรงของลำตัว
    - ตรวจสอบความลึกของการลง
    """
    required = ["RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_WRIST",
                "LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_WRIST",
                "RIGHT_HIP", "LEFT_HIP", "RIGHT_ANKLE", "LEFT_ANKLE"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.6):
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

    # 1. มุมศอก (ลงลึก = มุมเล็ก)
    R_angle = angle(R_sh, R_el, R_wr)
    L_angle = angle(L_sh, L_el, L_wr)
    avg_elbow = (R_angle + L_angle) / 2
    
    # ปรับให้ยืดหยุ่น: 70-170 degrees
    if avg_elbow < 90:
        elbow_score = 1.0  # ลงลึกมาก
    elif avg_elbow < 120:
        elbow_score = np.interp(avg_elbow, [90, 120], [1.0, 0.7])
    else:
        elbow_score = np.interp(avg_elbow, [120, 170], [0.7, 0.0])
    
    # 2. ความตรงของลำตัว (shoulder-hip-ankle)
    hip_y = (R_hip[1] + L_hip[1]) / 2
    ankle_y = (R_ankle[1] + L_ankle[1]) / 2
    sh_y = (R_sh[1] + L_sh[1]) / 2
    
    torso_len = abs(sh_y - hip_y) + 1e-6
    expected_hip_y = (sh_y + ankle_y) / 2
    deviation = abs(hip_y - expected_hip_y)
    
    straight_score = 1.0 - min(deviation / torso_len * 2, 1.0)
    
    # 3. ตรวจสอบว่าอยู่ในท่า plank position หรือไม่
    body_angle = angle(
        ((R_sh[0] + L_sh[0])/2, (R_sh[1] + L_sh[1])/2),
        ((R_hip[0] + L_hip[0])/2, (R_hip[1] + L_hip[1])/2),
        ((R_ankle[0] + L_ankle[0])/2, (R_ankle[1] + L_ankle[1])/2)
    )
    # ลำตัวควรตรง (angle ใกล้ 180)
    alignment_score = np.interp(body_angle, [160, 180], [0.7, 1.0])
    alignment_score = np.clip(alignment_score, 0, 1)
    
    final_score = (elbow_score * 0.5 + 
                   straight_score * 0.3 + 
                   alignment_score * 0.2)
    
    return float(np.clip(final_score, 0, 1))

def detect_plank(lm, mp):
    """ปรับปรุง Plank detection"""
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
    
    # ปรับให้ยืดหยุ่นขึ้น
    straight_score = 1.0 - min(deviation / torso_len * 1.5, 1.0)
    
    # ตรวจสอบว่าสะโพกไม่ห้อยต่ำหรือยกสูงเกินไป
    hip_drop = hip_y - expected_hip
    if abs(hip_drop) < torso_len * 0.1:  # ยอมรับได้ ±10%
        position_score = 1.0
    else:
        position_score = max(0.5, 1.0 - abs(hip_drop) / torso_len)
    
    final_score = (straight_score * 0.7 + position_score * 0.3)
    
    return float(np.clip(final_score, 0, 1))

def detect_situp(lm, mp):
    """
    Sit-up Detection ตามเกณฑ์:
    - ลำตัว (SHOULDER-HIP-KNEE): 40°-100°
    """
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
    
    # มุมลำตัว (SHOULDER-HIP-KNEE): 40°-100°
    R_torso_angle = angle(R_sh, R_hip, R_knee)
    L_torso_angle = angle(L_sh, L_hip, L_knee)
    avg_angle = (R_torso_angle + L_torso_angle) / 2
    
    # ยิ่งใกล้ 40° (นั่งขึ้นมาก) = คะแนนสูง
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
    """
    Lunge Detection ตามเกณฑ์:
    - เข่า (HIP-KNEE-ANKLE): 70°-140°
    - ลำตัว (SHOULDER-HIP-KNEE): 30°-90°
    """
    required = ["RIGHT_KNEE", "LEFT_KNEE",
                "RIGHT_HIP", "LEFT_HIP",
                "RIGHT_ANKLE", "LEFT_ANKLE",
                "RIGHT_SHOULDER", "LEFT_SHOULDER"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.4):
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
    
    # 1. มุมเข่า (HIP-KNEE-ANKLE): 70°-140°
    R_knee_angle = angle(R_hip, R_knee, R_ankle)
    L_knee_angle = angle(L_hip, L_knee, L_ankle)
    
    # เลือกขาที่งอมากกว่า (ขาหน้า)
    front_knee_angle = min(R_knee_angle, L_knee_angle)
    
    if 70 <= front_knee_angle <= 100:
        knee_score = 1.0
    elif 100 < front_knee_angle <= 140:
        knee_score = np.interp(front_knee_angle, [100, 140], [0.8, 0.2])
    else:
        knee_score = np.interp(front_knee_angle, [50, 70], [0.4, 1.0])
    
    # 2. มุมลำตัว (SHOULDER-HIP-KNEE): 30°-90°
    R_torso_angle = angle(R_sh, R_hip, R_knee)
    L_torso_angle = angle(L_sh, L_hip, L_knee)
    avg_torso = (R_torso_angle + L_torso_angle) / 2
    
    if 30 <= avg_torso <= 90:
        torso_score = 1.0
    elif avg_torso < 30:
        torso_score = np.interp(avg_torso, [10, 30], [0.3, 1.0])
    else:
        torso_score = np.interp(avg_torso, [90, 120], [1.0, 0.3])
    
    final_score = (knee_score * 0.6 + torso_score * 0.4)
    
    return float(np.clip(final_score, 0, 1))

def detect_dead_bug(lm, mp):
    """
    Dead Bug Detection ตามเกณฑ์:
    - แขน (SHOULDER-ELBOW-WRIST): 150°-180°
    - ขา (HIP-KNEE-ANKLE): 150°-180°
    """
    required = ["LEFT_WRIST", "RIGHT_WRIST",
                "LEFT_ELBOW", "RIGHT_ELBOW",
                "LEFT_ANKLE", "RIGHT_ANKLE",
                "LEFT_KNEE", "RIGHT_KNEE",
                "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.4):
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
    
    # 1. มุมแขน (SHOULDER-ELBOW-WRIST): 150°-180°
    R_arm_angle = angle(R_sh, R_elbow, R_wrist)
    L_arm_angle = angle(L_sh, L_elbow, L_wrist)
    avg_arm_angle = (R_arm_angle + L_arm_angle) / 2
    
    if 150 <= avg_arm_angle <= 180:
        arm_score = 1.0
    elif avg_arm_angle < 150:
        arm_score = np.interp(avg_arm_angle, [120, 150], [0.4, 1.0])
    else:
        arm_score = 0.5
    
    # 2. มุมขา (HIP-KNEE-ANKLE): 150°-180°
    R_leg_angle = angle(R_hip, R_knee, R_ankle)
    L_leg_angle = angle(L_hip, L_knee, L_ankle)
    avg_leg_angle = (R_leg_angle + L_leg_angle) / 2
    
    if 150 <= avg_leg_angle <= 180:
        leg_score = 1.0
    elif avg_leg_angle < 150:
        leg_score = np.interp(avg_leg_angle, [120, 150], [0.4, 1.0])
    else:
        leg_score = 0.5
    
    # 3. ตรวจสอบการเหยียดแขนขาขึ้น
    sh_y = (L_sh[1] + R_sh[1]) / 2
    wrist_y = (L_wrist[1] + R_wrist[1]) / 2
    ankle_y = (L_ankle[1] + R_ankle[1]) / 2
    
    arms_up = 1.0 if (sh_y - wrist_y) > 0.1 else 0.5
    legs_up = 1.0 if (sh_y - ankle_y) > 0.1 else 0.5
    
    final_score = (arm_score * 0.35 + leg_score * 0.35 + arms_up * 0.15 + legs_up * 0.15)
    
    return float(np.clip(final_score, 0, 1))

def detect_side_plank(lm, mp):
    """ปรับปรุง Side Plank detection"""
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
    
    # ลำตัวควรตรง: 165-180 องศา
    angle_score = np.interp(body_angle, [155, 180], [0.6, 1.0])
    angle_score = np.clip(angle_score, 0, 1)
    
    # ตรวจสอบว่าสะโพกไม่ห้อย
    expected_hip_y = (sh[1] + ankle[1]) / 2
    hip_drop = hip[1] - expected_hip_y
    
    if hip_drop < 0.02:  # สะโพกยกดี
        position_score = 1.0
    else:
        position_score = max(0.5, 1.0 - hip_drop * 5)
    
    final_score = (angle_score * 0.7 + position_score * 0.3)
    
    return float(np.clip(final_score, 0, 1))

def detect_russian_twist(lm, mp):
    """ปรับปรุง Russian Twist detection"""
    required = ["LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.6):
        return 0.0
    
    try:
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
    except Exception:
        return 0.0
    
    sh_center_x = (L_sh[0] + R_sh[0]) / 2
    hip_center_x = (L_hip[0] + R_hip[0]) / 2
    
    # การหมุนลำตัว
    twist = abs(sh_center_x - hip_center_x)
    twist_score = min(twist * 4, 1.0)  # ยิ่งหมุนมาก คะแนนยิ่งสูง
    
    # การโน้มตัวไปหลัง
    hip_y = (L_hip[1] + R_hip[1]) / 2
    sh_y = (L_sh[1] + R_sh[1]) / 2
    lean_back = sh_y - hip_y  # บวก = โน้มหลัง
    
    lean_score = np.clip(lean_back * 3, 0, 1)
    
    final_score = (twist_score * 0.6 + lean_score * 0.4)
    
    return float(np.clip(final_score, 0, 1))

def detect_lying_leg_raises(lm, mp):
    """
    Lying Leg Raises Detection ตามเกณฑ์:
    - สะโพก (SHOULDER-HIP-ANKLE): 30°-90°
    """
    required = ["LEFT_ANKLE", "RIGHT_ANKLE",
                "LEFT_HIP", "RIGHT_HIP",
                "LEFT_SHOULDER", "RIGHT_SHOULDER"]
    
    if not _check_landmarks_visible(lm, mp, required, 0.4):
        return 0.0
    
    try:
        L_ankle = landmark_xy(_get(lm, mp, "LEFT_ANKLE"))
        R_ankle = landmark_xy(_get(lm, mp, "RIGHT_ANKLE"))
        L_hip = landmark_xy(_get(lm, mp, "LEFT_HIP"))
        R_hip = landmark_xy(_get(lm, mp, "RIGHT_HIP"))
        L_sh = landmark_xy(_get(lm, mp, "LEFT_SHOULDER"))
        R_sh = landmark_xy(_get(lm, mp, "RIGHT_SHOULDER"))
    except Exception:
        return 0.0
    
    # มุมสะโพก (SHOULDER-HIP-ANKLE): 30°-90°
    R_hip_angle = angle(R_sh, R_hip, R_ankle)
    L_hip_angle = angle(L_sh, L_hip, L_ankle)
    avg_angle = (R_hip_angle + L_hip_angle) / 2
    
    # ยิ่งใกล้ 30° (ยกขาสูง) = คะแนนสูง
    if 30 <= avg_angle <= 60:
        score = 1.0
    elif 60 < avg_angle <= 90:
        score = np.interp(avg_angle, [60, 90], [0.8, 0.3])
    elif avg_angle < 30:
        score = np.interp(avg_angle, [10, 30], [0.5, 1.0])
    else:
        score = np.interp(avg_angle, [90, 120], [0.3, 0.0])
    
    # ตรวจสอบว่าขาทั้งสองยกพร้อมกัน
    ankle_diff = abs(L_ankle[1] - R_ankle[1])
    symmetry_score = 1.0 - min(ankle_diff * 3, 0.3)
    
    final_score = (score * 0.8 + symmetry_score * 0.2)
    
    return float(np.clip(final_score, 0, 1))