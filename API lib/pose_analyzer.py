# modules/pose_analyzer.py
import cv2
from .detectors import *
from .feedbacks import FEEDBACKS

class PoseAnalyzer:
    HOLD_POSES = {"Plank", "Side Plank"}
    REPS_POSES = {"Bodyweight Squat", "Push-ups", "Sit-ups",
                  "Lunge (Split Squat)", "Dead Bug", "Russian Twist", "Lying Leg Raises"}

    DETECTORS = {
        "Bodyweight Squat": detect_squat,
        "Push-ups": detect_pushup,
        "Plank": detect_plank,
        "Sit-ups": detect_situp,
        "Lunge (Split Squat)": detect_lunge,
        "Dead Bug": detect_dead_bug,
        "Side Plank": detect_side_plank,
        "Russian Twist": detect_russian_twist,
        "Lying Leg Raises": detect_lying_leg_raises,
    }

    def __init__(self, mp_pose):
        self.mp_pose = mp_pose
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose_detector.process(rgb)

    def detect(self, pose_name, landmarks):
        if not pose_name or pose_name not in self.DETECTORS:
            return 0.0
        # detector expects (landmarks, mp_pose)
        fn = self.DETECTORS[pose_name]
        try:
            val = fn(landmarks, self.mp_pose)
            return float(max(0.0, min(1.0, val)))
        except Exception:
            return 0.0

    def feedback(self, pose_name, landmarks, confidence, hold_time=0.0):
        fb = FEEDBACKS.get(pose_name)
        if not fb:
            return ""
        try:
            return fb(confidence, hold_time)
        except Exception:
            return ""
