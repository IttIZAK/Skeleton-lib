# main.py
import time
import json
import base64
import cv2
import numpy as np
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp

from modules.pose_analyzer import PoseAnalyzer
from modules.client_manager import ClientManager

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PoseAPI")

app = FastAPI(title="Pose Detection API", description="Real-time exercise pose detection", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

clients = ClientManager()
mp_pose = mp.solutions.pose
analyzer = PoseAnalyzer(mp_pose)

@app.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    await websocket.accept()
    client_id = clients.register(websocket.client.host)
    logger.info(f"[CONNECTED] {client_id}")

    try:
        async for message in websocket.iter_text():
            # command
            if message.startswith("{"):
                try:
                    cmd = json.loads(message)
                    pose = cmd.get("select_pose")
                    if pose:
                        clients.set_selected_pose(client_id, pose)
                        await websocket.send_json({"status": "pose_selected", "pose": pose})
                except Exception as e:
                    logger.error("cmd parse error: %s", e)
                continue

            # decode image
            try:
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(message), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Frame decode failed")
            except Exception as e:
                await websocket.send_json({"error": "decode_failed", "detail": str(e)})
                continue

            ts = time.time()
            results = analyzer.process_frame(frame)
            selected_pose = clients.get_pose(client_id)
            response = clients.make_response(client_id, selected_pose, ts)

            if results.pose_landmarks and selected_pose:
                landmarks = results.pose_landmarks.landmark
                confidence = analyzer.detect(selected_pose, landmarks)
                clients.update_counters(client_id, selected_pose, confidence, ts)
                hold = clients.get_hold_time(client_id, selected_pose)["current"]
                advice_msg = analyzer.feedback(selected_pose, landmarks, confidence, hold)

                response.update({
                    "confidence": round(float(confidence), 3),
                    "advice": advice_msg,
                    "reps": clients.clients[client_id].reps_counts.copy(),
                    "holds": clients.clients[client_id].hold_times.copy()
                })
            else:
                # Always send advice (or prompt) so Flutter can update UI
                response.update({
                    "confidence": 0.0,
                    "advice": "กรุณาเข้ามาในกรอบกล้อง" if not results.pose_landmarks else "กรุณาเลือกท่าที่ต้องการออกกำลังกาย",
                    "reps": clients.clients[client_id].reps_counts.copy(),
                    "holds": clients.clients[client_id].hold_times.copy()
                })

            await websocket.send_json(response)

    except WebSocketDisconnect:
        clients.remove(client_id)
        logger.info(f"[DISCONNECTED] {client_id}")
    except Exception as e:
        logger.error(f"[UNEXPECTED ERROR] {e}")
        clients.remove(client_id)

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Pose Detection API",
        "version": "3.0.0",
        "available_poses": list(PoseAnalyzer.DETECTORS.keys()),
        "active_clients": clients.count()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "active_clients": clients.count(), "timestamp": time.time()}

@app.get("/poses")
async def list_poses():
    return {
        "hold_poses": list(PoseAnalyzer.HOLD_POSES),
        "rep_poses": list(PoseAnalyzer.REPS_POSES),
        "all_poses": list(PoseAnalyzer.DETECTORS.keys())
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Pose Detection API v3...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

#python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000