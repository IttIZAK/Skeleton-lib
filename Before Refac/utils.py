import numpy as np
import math

def normalize_landmarks(landmarks):
    """
    Normalize landmark ให้ scale อยู่ระหว่าง 0-1 โดยอิงตามความสูงร่างกาย
    landmarks: list ของ (x,y,z,visibility)
    """
    landmarks = np.array(landmarks)
    min_vals = np.min(landmarks[:, :3], axis=0)
    max_vals = np.max(landmarks[:, :3], axis=0)
    normalized = (landmarks[:, :3] - min_vals) / (max_vals - min_vals + 1e-6)
    return normalized.tolist()

def calculate_angle(a, b, c):
    """
    คำนวณมุมระหว่าง 3 จุด a,b,c (เช่น ไหล่-ศอก-ข้อมือ)
    แต่ละจุดเป็น tuple (x,y)
    """
    a = np.array(a[:2])  # ใช้เฉพาะ x,y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle
