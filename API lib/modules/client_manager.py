# modules/client_manager.py
import time

class Client:
    def __init__(self, cid):
        self.cid = cid
        self.selected_pose = None
        self.reps_counts = {}
        self.hold_times = {}
        self.last_ts = time.time()
        
        # State tracking for each pose - ใช้ระบบ peak detection แบบง่าย
        self.pose_states = {}  # {pose_name: "high" | "low"}
        self.last_confidence = {}  # {pose_name: float}
        self.last_rep_time = {}  # {pose_name: timestamp}
        self.peak_detected = {}  # {pose_name: bool} - สำหรับตรวจจับจุดสูงสุด

class ClientManager:
    # Cooldown periods (seconds) - ลดลงให้นับได้เร็วขึ้น
    COOLDOWN = {
        "Bodyweight Squat": 0.5,
        "Push-ups": 0.4,
        "Sit-ups": 0.5,
        "Lunge (Split Squat)": 0.6,
        "Dead Bug": 0.7,
        "Russian Twist": 0.3,
        "Lying Leg Raises": 0.5,
    }
    
    # Threshold สำหรับการตรวจจับ peak (จุดสูงสุด) และ valley (จุดต่ำสุด)
    HIGH_THRESHOLD = 0.70   # ต้องสูงกว่านี้ถึงจะถือว่าทำท่าได้ดี (peak)
    LOW_THRESHOLD = 0.35    # ต้องต่ำกว่านี้ถึงจะถือว่ากลับสู่ตำแหน่งเริ่มต้น (valley)
    
    # สำหรับท่าค้าง
    HOLD_THRESHOLD = 0.80

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
                client.pose_states[pose] = "low"  # เริ่มที่ low
                client.last_confidence[pose] = 0.0
                client.last_rep_time[pose] = 0.0
                client.peak_detected[pose] = False

    def get_pose(self, cid):
        return self.clients.get(cid).selected_pose if cid in self.clients else None

    def get_hold_time(self, cid, pose):
        client = self.clients.get(cid)
        if not client:
            return {"current": 0.0, "best": 0.0}
        return client.hold_times.get(pose, {"current": 0.0, "best": 0.0})

    def _check_cooldown(self, client, pose, ts):
        """ตรวจสอบว่าพ้น cooldown แล้วหรือยัง"""
        cooldown = self.COOLDOWN.get(pose, 0.4)
        last_time = client.last_rep_time.get(pose, 0.0)
        return (ts - last_time) >= cooldown

    def update_counters(self, cid, pose, confidence, ts):
        client = self.clients.get(cid)
        if not client:
            return
        
        dt = ts - client.last_ts if client.last_ts else 0.0
        client.last_ts = ts

        # Initialize states if needed
        if pose not in client.pose_states:
            client.pose_states[pose] = "low"
            client.last_confidence[pose] = 0.0
            client.last_rep_time[pose] = 0.0
            client.peak_detected[pose] = False

        # Hold-based poses (Plank, Side Plank)
        if pose in ["Plank", "Side Plank"]:
            hold = client.hold_times.get(pose, {"current": 0.0, "best": 0.0})
            
            if confidence > self.HOLD_THRESHOLD:
                hold["current"] = hold.get("current", 0.0) + dt
            else:
                # Reset if form breaks
                if hold["current"] > 0.3:  # บันทึกถ้าค้างได้อย่างน้อย 0.3 วินาที
                    hold["best"] = max(hold.get("best", 0.0), hold["current"])
                hold["current"] = 0.0
            
            client.hold_times[pose] = hold
            client.pose_states[pose] = "holding" if confidence > self.HOLD_THRESHOLD else "low"
            client.last_confidence[pose] = confidence
            return

        # Rep-based poses - ใช้ Peak Detection แบบง่าย
        # Logic: low → high (ตรวจจับ peak) → low → count!
        
        current_state = client.pose_states[pose]
        last_conf = client.last_confidence[pose]
        peak_detected = client.peak_detected[pose]

        # ตรวจจับว่าอยู่ในสถานะไหน
        is_high = confidence >= self.HIGH_THRESHOLD
        is_low = confidence <= self.LOW_THRESHOLD

        if current_state == "low":
            # รอให้ไปถึงจุดสูง (peak)
            if is_high:
                client.pose_states[pose] = "high"
                client.peak_detected[pose] = True  # mark ว่าเจอ peak แล้ว
                
        elif current_state == "high":
            # อยู่ที่จุดสูงแล้ว รอให้กลับมาต่ำ
            if is_low and peak_detected:
                # กลับมาต่ำแล้ว = นับ 1 rep!
                if self._check_cooldown(client, pose, ts):
                    client.reps_counts[pose] = client.reps_counts.get(pose, 0) + 1
                    client.last_rep_time[pose] = ts
                    # print(f"[REP COUNT] {pose}: {client.reps_counts[pose]}")  # debug
                
                # Reset state
                client.pose_states[pose] = "low"
                client.peak_detected[pose] = False
            elif not is_high and not is_low:
                # อยู่ระหว่างกลาง ยังคงเป็น high
                pass

        # Store last confidence for debugging
        client.last_confidence[pose] = confidence

    def get_state_debug(self, cid, pose):
        """สำหรับ debug - ดูสถานะปัจจุบัน"""
        client = self.clients.get(cid)
        if not client or pose not in client.pose_states:
            return "N/A"
        state = client.pose_states[pose]
        peak = "✓" if client.peak_detected.get(pose, False) else "✗"
        return f"{state} (peak:{peak})"

    def make_response(self, cid, pose, ts):
        client = self.clients.get(cid)
        if not client:
            return {"status": "error", "message": "client not found"}

        hold_info = {}
        if pose in ["Plank", "Side Plank"]:
            hold_info = {pose: client.hold_times.get(pose, {"current": 0.0, "best": 0.0})}

        # Add debug info
        state = client.pose_states.get(pose, "waiting") if pose else "N/A"
        last_conf = client.last_confidence.get(pose, 0.0) if pose else 0.0

        return {
            "status": "ok",
            "pose": pose or "N/A",
            "confidence": 0.0,
            "advice": "",
            "reps": client.reps_counts.copy(),
            "holds": hold_info,
            "state": state,  # เพิ่มสถานะปัจจุบันให้ Flutter ดู
            "last_conf": round(last_conf, 2)  # เพิ่ม confidence ล่าสุดเพื่อ debug
        }