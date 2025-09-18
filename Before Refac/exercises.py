from utils import calculate_angle

class ExerciseDetector:
    def __init__(self):
        self.counters = {
            "squat": 0,
            "pushup": 0,
            "situp": 0,
            "plank": 0,
            "forward_lunge": 0,
            "dead_bug": 0,
            "side_plank": 0,
            "russian_twist": 0,
            "lying_leg_raises": 0,
        }
        self.status = {k: None for k in self.counters.keys()}

    def detect(self, landmarks):
        """
        ตรวจจับท่าและอัปเดต counter
        landmarks: list ของ (x,y,z,visibility)
        return: (pose_name, reps_count)
        """
        pose_detected = "N/A"

        # ✅ Squat example
        left_knee_angle = calculate_angle(
            landmarks[23], landmarks[25], landmarks[27]  # hip-knee-ankle
        )
        if left_knee_angle < 90:
            if self.status["squat"] != "down":
                self.status["squat"] = "down"
        else:
            if self.status["squat"] == "down":
                self.counters["squat"] += 1
                self.status["squat"] = "up"
                pose_detected = "squat"

        # ✅ Push-up example
        left_elbow_angle = calculate_angle(
            landmarks[11], landmarks[13], landmarks[15]  # shoulder-elbow-wrist
        )
        if left_elbow_angle < 90:
            if self.status["pushup"] != "down":
                self.status["pushup"] = "down"
        else:
            if self.status["pushup"] == "down":
                self.counters["pushup"] += 1
                self.status["pushup"] = "up"
                pose_detected = "pushup"

        # TODO: เพิ่ม logic สำหรับท่าอื่น ๆ

        return pose_detected, self.counters
