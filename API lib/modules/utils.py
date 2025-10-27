# modules/utils.py
import numpy as np
import math

def landmark_xy(lm):
    """Return (x, y) from a mediapipe landmark object."""
    return (lm.x, lm.y)

def angle(a, b, c):
    """Compute angle ABC in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def check_visibility(lm, threshold=0.5):
    """Return true if lm has visibility > threshold or no visibility attr."""
    return lm.visibility > threshold if hasattr(lm, "visibility") else True
