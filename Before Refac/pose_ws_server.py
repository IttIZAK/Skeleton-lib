import base64
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket
import json

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.websocket("/ws/pose")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            data = await ws.receive_text()
            frame_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            pose_name = "N/A"
            confidence = 0.0
            reps = 0
            landmarks = []

            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    landmarks.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "v": lm.visibility
                    })

                pose_name = "squat"   # TODO: ใส่ logic ตรวจจับท่า
                confidence = 0.9
                reps = 3

            response = {
                "pose": pose_name,
                "confidence": confidence,
                "reps": reps,
                "landmarks": landmarks
            }

            await ws.send_text(json.dumps(response))

        except Exception as e:
            print("Error:", e)
            break

##python -m uvicorn pose_ws_server:app --reload --host 0.0.0.0 --port 8000