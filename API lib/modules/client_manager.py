# modules/client_manager.py
import time

class Client:
    def __init__(self, cid):
        self.cid = cid
        self.selected_pose = None
        self.reps_counts = {}
        self.hold_times = {}
        self.last_ts = time.time()
        
        # State tracking for each pose - ใช้ระบบ peak detection
        self.pose_states = {}  # {pose_name: "high" | "low" | "transition"}
        self.last_confidence = {}  # {pose_name: float}
        self.last_rep_time = {}  # {pose_name: timestamp}
        self.peak_detected = {}  # {pose_name: bool} - สำหรับตรวจจับจุดสูงสุด
        self.confidence_history = {}  # {pose_name: [conf1, conf2, ...]} - เก็บประวัติ 5 frame

class ClientManager:
    # Cooldown periods (seconds) - เพิ่มเวลาให้เหมาะสม
    COOLDOWN = {
        "Bodyweight Squat": 0.8,     # เพิ่มจาก 0.6
        "Push-ups": 0.7,              # เพิ่มจาก 0.5
        "Sit-ups": 0.8,               # เพิ่มจาก 0.6
        "Lunge (Split Squat)": 0.9,  # เพิ่มจาก 0.7
        "Dead Bug": 1.0,              # เพิ่มจาก 0.8
        "Russian Twist": 0.5,         # เพิ่มจาก 0.4
        "Lying Leg Raises": 0.8,      # เพิ่มจาก 0.6
    }
    
    # Threshold สำหรับการตรวจจับ
    HIGH_THRESHOLD = 0.50   # ลดจาก 0.55 ให้ตรวจจับง่ายขึ้น
    LOW_THRESHOLD = 0.30    # ✅ เพิ่มขึ้นจาก 0.25 เป็น 0.30
    TRANSITION_THRESHOLD = 0.38  # ลดจาก 0.42
    
    # สำหรับท่าค้าง
    HOLD_THRESHOLD = 0.55   # ลดจาก 0.60
    HOLD_MIN_DURATION = 0.3

    def __init__(self):
        self.clients = {}

    def register(self, host):
        cid = f"{host}_{int(time.time()*1000)}"
        self.clients[cid] = Client(cid)
        return cid

    def remove(self, cid):
        self.clients.pop(cid, None)

    def count(self):
        return len(self.clients)

    def stats(self):
        return {
            cid: {"pose": c.selected_pose, "reps": c.reps_counts, "holds": c.hold_times}
            for cid, c in self.clients.items()
        }

    def set_selected_pose(self, cid, pose):
        if cid in self.clients:
            client = self.clients[cid]
            client.selected_pose = pose
            # Initialize state for new pose
            if pose not in client.pose_states:
                client.pose_states[pose] = "low"
                client.last_confidence[pose] = 0.0
                client.last_rep_time[pose] = 0.0
                client.peak_detected[pose] = False
                client.confidence_history[pose] = []
                # ✅ Initialize counters
                if pose not in client.reps_counts:
                    client.reps_counts[pose] = 0
                if pose not in client.hold_times:
                    client.hold_times[pose] = {"current": 0.0, "best": 0.0}

    def get_pose(self, cid):
        return self.clients.get(cid).selected_pose if cid in self.clients else None

    def get_hold_time(self, cid, pose):
        client = self.clients.get(cid)
        if not client:
            return {"current": 0.0, "best": 0.0}
        return client.hold_times.get(pose, {"current": 0.0, "best": 0.0})

    def _check_cooldown(self, client, pose, ts):
        """ตรวจสอบว่าพ้น cooldown แล้วหรือยัง"""
        cooldown = self.COOLDOWN.get(pose, 0.7)  # default เพิ่มเป็น 0.7
        last_time = client.last_rep_time.get(pose, 0.0)
        return (ts - last_time) >= cooldown

    def _smooth_confidence(self, client, pose, confidence):
        """ทำให้ confidence นุ่มนวลขึ้นด้วยการเฉลี่ย"""
        if pose not in client.confidence_history:
            client.confidence_history[pose] = []
        
        history = client.confidence_history[pose]
        history.append(confidence)
        
        # เก็บแค่ 2 frame ล่าสุด
        if len(history) > 2:
            history.pop(0)
        
        return sum(history) / len(history)

    def update_counters(self, cid, pose, confidence, ts, full_body_visible=True):
        """
        อัพเดท counters และ hold times
        
        ✅ แก้ไข: ทำให้ชัดเจนว่าเมื่อไรนับ rep
        """
        client = self.clients.get(cid)
        if not client:
            return
        
        # ⚠️ CRITICAL: ถ้าเห็นไม่เต็มตัว -> ไม่นับเลย!
        if not full_body_visible:
            # Reset hold time สำหรับท่าค้าง
            if pose in ["Plank", "Side Plank"]:
                hold = client.hold_times.get(pose, {"current": 0.0, "best": 0.0})
                if hold["current"] > self.HOLD_MIN_DURATION:
                    hold["best"] = max(hold.get("best", 0.0), hold["current"])
                hold["current"] = 0.0
                client.hold_times[pose] = hold
            
            # Reset state
            client.pose_states[pose] = "low"
            client.peak_detected[pose] = False
            client.last_confidence[pose] = 0.0
            return
        
        dt = ts - client.last_ts if client.last_ts else 0.0
        client.last_ts = ts

        # Initialize states if needed
        if pose not in client.pose_states:
            client.pose_states[pose] = "low"
            client.last_confidence[pose] = 0.0
            client.last_rep_time[pose] = 0.0
            client.peak_detected[pose] = False
            client.confidence_history[pose] = []
            if pose not in client.reps_counts:
                client.reps_counts[pose] = 0

        # Smooth confidence เพื่อลด noise
        smoothed_conf = self._smooth_confidence(client, pose, confidence)

        # Hold-based poses (Plank, Side Plank)
        if pose in ["Plank", "Side Plank"]:
            hold = client.hold_times.get(pose, {"current": 0.0, "best": 0.0})
            
            if full_body_visible and smoothed_conf > self.HOLD_THRESHOLD:
                hold["current"] = hold.get("current", 0.0) + dt
                client.pose_states[pose] = "holding"
            else:
                if hold["current"] > self.HOLD_MIN_DURATION:
                    hold["best"] = max(hold.get("best", 0.0), hold["current"])
                hold["current"] = 0.0
                client.pose_states[pose] = "not_holding"
            
            client.hold_times[pose] = hold
            client.last_confidence[pose] = smoothed_conf
            return

        # ✅ Rep-based poses - ใช้ทั้ง raw และ smoothed confidence
        current_state = client.pose_states[pose]
        
        # ใช้ raw confidence สำหรับการตัดสินใจที่รวดเร็ว
        use_raw_for_low = confidence <= self.LOW_THRESHOLD
        use_smoothed_for_high = smoothed_conf >= self.HIGH_THRESHOLD

        # State Machine Logic
        if current_state == "low":
            if use_smoothed_for_high:
                # เข้าสู่ท่าที่ดี -> เปลี่ยนเป็น high
                client.pose_states[pose] = "high"
                client.peak_detected[pose] = True
                print(f"[{pose}] LOW -> HIGH (raw: {confidence:.2f}, smoothed: {smoothed_conf:.2f})")
                
        elif current_state == "high":
            # ✅ ใช้ raw confidence เพื่อให้ตอบสนองเร็วขึ้น
            if use_raw_for_low and client.peak_detected[pose]:
                # กลับสู่ตำแหน่งเริ่มต้น + เคยผ่าน peak -> นับ rep!
                if self._check_cooldown(client, pose, ts):
                    client.reps_counts[pose] = client.reps_counts.get(pose, 0) + 1
                    client.last_rep_time[pose] = ts
                    print(f"[{pose}] ✅ REP COUNTED! Total: {client.reps_counts[pose]} (raw: {confidence:.2f})")
                else:
                    print(f"[{pose}] ⏳ Cooldown not passed (last rep: {ts - client.last_rep_time[pose]:.2f}s ago)")
                
                client.pose_states[pose] = "low"
                client.peak_detected[pose] = False
                print(f"[{pose}] HIGH -> LOW (returning to start position)")

        # Store last confidence
        client.last_confidence[pose] = smoothed_conf

    def get_state_debug(self, cid, pose):
        """สำหรับ debug - ดูสถานะปัจจุบัน"""
        client = self.clients.get(cid)
        if not client or pose not in client.pose_states:
            return "N/A"
        
        state = client.pose_states[pose]
        peak = "✓" if client.peak_detected.get(pose, False) else "✗"
        conf = client.last_confidence.get(pose, 0.0)
        reps = client.reps_counts.get(pose, 0)
        
        return f"{state} (peak:{peak}, conf:{conf:.2f}, reps:{reps})"

    def make_response(self, cid, pose, ts):
        client = self.clients.get(cid)
        if not client:
            return {"status": "error", "message": "client not found"}

        # ✅ สร้าง holds ในรูปแบบที่ Flutter คาดหวัง
        hold_info = {}
        if pose in ["Plank", "Side Plank"]:
            hold_data = client.hold_times.get(pose, {"current": 0.0, "best": 0.0})
            hold_info = {
                pose: {
                    "current_hold": hold_data.get("current", 0.0),  # ✅ เปลี่ยนชื่อให้ตรง
                    "best_hold": hold_data.get("best", 0.0)         # ✅ เปลี่ยนชื่อให้ตรง
                }
            }

        state = client.pose_states.get(pose, "waiting") if pose else "N/A"
        last_conf = client.last_confidence.get(pose, 0.0) if pose else 0.0

        return {
            "status": "ok",
            "pose": pose or "N/A",
            "confidence": 0.0,
            "advice": "",
            "reps": client.reps_counts.copy(),  # ✅ ส่งทุก pose
            "holds": hold_info,
            "state": state,
            "last_conf": round(last_conf, 2)
        }