# client_manager_fixed_full_v4.py
# ✅ รวม logic ทั้งหมด (Hold + On_Peak + Continuous + Direction Twist)
# ✅ Dead Bug / Lying Leg Raises / Push-ups => Continuous (conf > 0.75)
# ✅ Russian Twist => Direction twist detection
# ✅ ท่าอื่นๆ เดิมครบ (Squat, Sit-up, Lunge, Plank, Side Plank)

import time

class Client:
    def __init__(self, cid):
        self.cid = cid
        self.selected_pose = None
        self.reps_counts = {}
        self.hold_times = {}
        self.last_ts = time.time()
        self.pose_states = {}
        self.last_confidence = {}
        self.last_rep_time = {}
        self.peak_detected = {}
        self.confidence_history = {}
        self.twist_direction = "center"  # สำหรับ Russian Twist


class ClientManager:
    COOLDOWN = {
        "Bodyweight Squat": 0.8,
        "Push-ups": 2,  # ปรับให้เร็วขึ้นเล็กน้อย
        "Sit-ups": 0.8,
        "Lunge (Split Squat)": 0.8,
        "Dead Bug": 1.9,
        "Russian Twist": 0.5,
        "Lying Leg Raises": 1,
    }

    POSE_THRESHOLDS = {
        "Bodyweight Squat": {
            "high": 0.50, "low": 0.30, "smooth_frames": 2, "count_mode": "peak_to_low"
        },
        # ✅ Push-ups: เปลี่ยนเป็น continuous mode
        "Push-ups": {
            "high": 0.75, "continuous": True, "use_raw": True
        },
        "Sit-ups": {
            "high": 0.45, "low": 0.28, "smooth_frames": 2, "count_mode": "peak_to_low"
        },
        "Lunge (Split Squat)": {
            "high": 0.35, "low": 0.20, "smooth_frames": 2, "count_mode": "on_peak"
        },
        "Plank": {
            "high": 0.55, "low": 0.35, "smooth_frames": 3, "count_mode": "hold"
        },
        "Side Plank": {
            "high": 0.55, "low": 0.35, "smooth_frames": 3, "count_mode": "hold"
        },
        "Dead Bug": {
            "high": 0.75, "continuous": True, "use_raw": True
        },
        "Lying Leg Raises": {
            "high": 0.75, "continuous": True, "use_raw": True
        },
        "Russian Twist": {
            "high": 0.30, "low": 0.10, "smooth_frames": 1,
            "count_mode": "direction_twist", "use_raw": True, "angle_tolerance": 0.10
        },
    }

    DEFAULT_THRESHOLD = {"high": 0.45, "low": 0.28, "smooth_frames": 2}
    HOLD_THRESHOLD = 0.55
    HOLD_MIN_DURATION = 0.3

    def __init__(self):
        self.clients = {}
    
    def get_pose(self, cid):
        """Return current selected pose of the client"""
        client = self.clients.get(cid)
        if client:
            return client.selected_pose
        return None

    def set_selected_pose(self, cid, pose):
        """Set the selected pose for the client"""
        if cid in self.clients:
            client = self.clients[cid]
            client.selected_pose = pose
            if pose not in client.reps_counts:
                client.reps_counts[pose] = 0
            if pose not in client.pose_states:
                client.pose_states[pose] = "low"
            if pose not in client.last_confidence:
                client.last_confidence[pose] = 0.0
            if pose not in client.last_rep_time:
                client.last_rep_time[pose] = 0.0
            if pose not in client.confidence_history:
                client.confidence_history[pose] = []
            if pose not in client.hold_times:
                client.hold_times[pose] = {"current": 0.0, "best": 0.0}

    def get_hold_time(self, cid, pose):
        """Return current and best hold times"""
        client = self.clients.get(cid)
        if not client:
            return {"current": 0.0, "best": 0.0}
        return client.hold_times.get(pose, {"current": 0.0, "best": 0.0})
    
    def register(self, host):
        """ลงทะเบียน client ใหม่"""
        cid = f"{host}_{int(time.time() * 1000)}"
        self.clients[cid] = Client(cid)
        return cid

    def remove(self, cid):
        """ลบ client ออกจากระบบ"""
        if cid in self.clients:
            del self.clients[cid]

    # --- Utility functions for main.py ---
    def get_pose(self, cid):
        client = self.clients.get(cid)
        if client:
            return client.selected_pose
        return None

    def set_selected_pose(self, cid, pose):
        if cid in self.clients:
            client = self.clients[cid]
            client.selected_pose = pose
            if pose not in client.reps_counts:
                client.reps_counts[pose] = 0
            if pose not in client.pose_states:
                client.pose_states[pose] = "low"
            if pose not in client.last_confidence:
                client.last_confidence[pose] = 0.0
            if pose not in client.last_rep_time:
                client.last_rep_time[pose] = 0.0
            if pose not in client.confidence_history:
                client.confidence_history[pose] = []
            if pose not in client.hold_times:
                client.hold_times[pose] = {"current": 0.0, "best": 0.0}

    def get_hold_time(self, cid, pose):
        client = self.clients.get(cid)
        if not client:
            return {"current": 0.0, "best": 0.0}
        return client.hold_times.get(pose, {"current": 0.0, "best": 0.0})

    # --- Internal helpers ---
    def _get_thresholds(self, pose):
        return self.POSE_THRESHOLDS.get(pose, self.DEFAULT_THRESHOLD)

    def _check_cooldown(self, client, pose, ts):
        cooldown = self.COOLDOWN.get(pose, 0.7)
        last_time = client.last_rep_time.get(pose, 0)
        return (ts - last_time) >= cooldown

    def _smooth_confidence(self, client, pose, confidence):
        thresholds = self._get_thresholds(pose)
        if thresholds.get("use_raw", False):
            return confidence
        max_frames = thresholds.get("smooth_frames", 2)
        if pose not in client.confidence_history:
            client.confidence_history[pose] = []
        hist = client.confidence_history[pose]
        hist.append(confidence)
        if len(hist) > max_frames:
            hist.pop(0)
        return sum(hist) / len(hist)

    # --- Core update logic ---
    def update_counters(self, cid, pose, confidence, ts, full_body_visible=True):
        client = self.clients.get(cid)
        if not client or not pose:
            return

        thresholds = self._get_thresholds(pose)
        high = thresholds.get("high", 0.5)
        low = thresholds.get("low", 0.3)
        count_mode = thresholds.get("count_mode", "peak_to_low")
        continuous = thresholds.get("continuous", False)
        use_raw = thresholds.get("use_raw", False)
        conf = confidence if use_raw else self._smooth_confidence(client, pose, confidence)

        # (1) Hold mode
        if count_mode == "hold":
            hold = client.hold_times.get(pose, {"current": 0.0, "best": 0.0})
            dt = ts - client.last_ts
            if full_body_visible and conf > self.HOLD_THRESHOLD:
                hold["current"] += dt
            else:
                if hold["current"] > self.HOLD_MIN_DURATION:
                    hold["best"] = max(hold["best"], hold["current"])
                hold["current"] = 0.0
            client.hold_times[pose] = hold
            client.last_confidence[pose] = conf
            client.last_ts = ts
            return

        # (2) Continuous mode (Dead Bug, Leg Raises, Push-ups)
        if continuous:
            if conf >= high and self._check_cooldown(client, pose, ts):
                client.reps_counts[pose] = client.reps_counts.get(pose, 0) + 1
                client.last_rep_time[pose] = ts
                print(f"[{pose}] 🔁 Continuous REP #{client.reps_counts[pose]} (conf: {conf:.2f})")
            client.last_confidence[pose] = conf
            client.last_ts = ts
            return

        # (3) Direction twist mode (Russian Twist)
        if count_mode == "direction_twist":
            tol = thresholds.get("angle_tolerance", 0.1)
            direction_now = "center"
            if conf > high + tol:
                direction_now = "right"
            elif conf < low - tol:
                direction_now = "left"

            if direction_now != client.twist_direction and direction_now in ("left", "right"):
                if self._check_cooldown(client, pose, ts):
                    client.reps_counts[pose] = client.reps_counts.get(pose, 0) + 1
                    client.last_rep_time[pose] = ts
                    print(f"[{pose}] 🔄 Twist {direction_now} → REP #{client.reps_counts[pose]} ({conf:.2f})")
            client.twist_direction = direction_now
            client.last_confidence[pose] = conf
            client.last_ts = ts
            return

        # (4) Default (Squat, Sit-up, Lunge)
        current_state = client.pose_states.get(pose, "low")
        if count_mode == "on_peak":
            if current_state == "low" and conf >= high:
                if self._check_cooldown(client, pose, ts):
                    client.reps_counts[pose] = client.reps_counts.get(pose, 0) + 1
                    client.last_rep_time[pose] = ts
                    print(f"[{pose}] ✅ ON_PEAK REP #{client.reps_counts[pose]} ({conf:.2f})")
                client.pose_states[pose] = "high"
            elif current_state == "high" and conf < low:
                client.pose_states[pose] = "low"
        else:
            if current_state == "low" and conf >= high:
                client.pose_states[pose] = "high"
            elif current_state == "high" and conf < low:
                if self._check_cooldown(client, pose, ts):
                    client.reps_counts[pose] = client.reps_counts.get(pose, 0) + 1
                    client.last_rep_time[pose] = ts
                    print(f"[{pose}] ✅ PEAK_TO_LOW REP #{client.reps_counts[pose]} ({conf:.2f})")
                client.pose_states[pose] = "low"

        client.last_confidence[pose] = conf
        client.last_ts = ts

    def get_state_debug(self, cid, pose):
        c = self.clients.get(cid)
        if not c:
            return "N/A"
        conf = c.last_confidence.get(pose, 0)
        reps = c.reps_counts.get(pose, 0)
        state = c.pose_states.get(pose, "low")
        return f"{pose}: {state}, conf={conf:.2f}, reps={reps}"

    def make_response(self, cid, pose, ts):
        c = self.clients.get(cid)
        if not c:
            return {"status": "error", "message": "client not found"}
        return {
            "status": "ok",
            "pose": pose,
            "reps": c.reps_counts,
            "last_conf": round(c.last_confidence.get(pose, 0), 2),
            "state": c.pose_states.get(pose, "low")
        }
