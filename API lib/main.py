# main.py - DEBUG VERSION
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
from modules.utils import check_full_body_visible, check_pose_specific_visibility

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PoseAPI")

app = FastAPI(
    title="Pose Detection API",
    description="Real-time exercise pose detection",
    version="3.2.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Managers
clients = ClientManager()
mp_pose = mp.solutions.pose
analyzer = PoseAnalyzer(mp_pose)

# ---------------- WebSocket ----------------
@app.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    await websocket.accept()
    client_id = clients.register(websocket.client.host)
    logger.info(f"[CONNECTED] {client_id}")

    try:
        async for message in websocket.iter_text():
            ts = time.time()

            # ---------------- Command ----------------
            if message.startswith("{"):
                try:
                    cmd = json.loads(message)
                    pose = cmd.get("select_pose")
                    if pose:
                        clients.set_selected_pose(client_id, pose)
                        logger.info(f"[{client_id}] Selected pose: {pose}")
                        await websocket.send_json({"status": "pose_selected", "pose": pose})
                except Exception as e:
                    logger.error("cmd parse error: %s", e)
                continue

            # ---------------- Image ----------------
            try:
                frame = cv2.imdecode(
                    np.frombuffer(base64.b64decode(message), np.uint8),
                    cv2.IMREAD_COLOR
                )
                if frame is None:
                    raise ValueError("Frame decode failed")
            except Exception as e:
                await websocket.send_json({"error": "decode_failed", "detail": str(e)})
                continue

            selected_pose = clients.get_pose(client_id)
            
            # ✅ สร้าง response พื้นฐานที่มี reps และ holds เสมอ
            client = clients.clients.get(client_id)
            response = {
                "status": "ok",
                "pose": selected_pose or "N/A",
                "confidence": 0.0,
                "advice": "",
                "reps": client.reps_counts.copy() if client else {},  # ✅ ส่งเสมอ
                "holds": {},
                "state": "waiting",
                "last_conf": 0.0,
                "visibility_score": 0.0,
                "full_body_visible": False,
                "ready_to_start": False
            }

            if results := analyzer.process_frame(frame):
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # ตรวจสอบว่าเห็นร่างกายเต็มตัวหรือไม่
                    full_body_visible, missing_parts, visibility_score = check_full_body_visible(
                        landmarks, analyzer.mp_pose, min_visibility=0.5
                    )
                    
                    # ถ้ายังไม่มีท่าที่เลือก -> แจ้งเตือน
                    if not selected_pose:
                        response.update({
                            "confidence": 0.0,
                            "advice": "กรุณาเลือกท่าที่ต้องการออกกำลังกาย",
                            "reps": client.reps_counts.copy() if client else {},
                            "holds": {},
                            "state": "waiting_pose_selection",
                            "last_conf": 0.0,
                            "visibility_score": round(visibility_score, 2),
                            "full_body_visible": full_body_visible,
                            "ready_to_start": False
                        })
                    elif not full_body_visible:
                        # เห็นไม่ครบ -> ให้ confidence ต่ำ (0-20%) และไม่นับ
                        partial_conf = min(visibility_score * 0.20, 0.20)
                        missing_text = ", ".join(missing_parts[:3])
                        response.update({
                            "confidence": round(partial_conf, 3),
                            "advice": f"!! ถอยออกให้เห็นร่างกายเต็มตัว (ขาด: {missing_text})",
                            "reps": client.reps_counts.copy() if client else {},
                            "holds": {},
                            "state": "body_not_visible",
                            "last_conf": round(partial_conf, 2),
                            "visibility_score": round(visibility_score, 2),
                            "full_body_visible": False,
                            "missing_parts": missing_parts,
                            "ready_to_start": False
                        })
                    else:
                        # เห็นร่างกายเต็มตัวแล้ว -> ตรวจสอบท่าเฉพาะ
                        pose_visible, pose_missing, pose_vis_score = check_pose_specific_visibility(
                            landmarks, analyzer.mp_pose, selected_pose, min_visibility=0.5
                        )
                        
                        if not pose_visible:
                            # จุดสำคัญของท่านี้มองไม่เห็นครบ -> ให้ confidence ต่ำ
                            partial_conf = min(pose_vis_score * 0.20, 0.20)
                            response.update({
                                "confidence": round(partial_conf, 3),
                                "advice": f"!! ปรับมุมกล้องให้เห็นท่า {selected_pose} ชัดเจนขึ้น",
                                "reps": client.reps_counts.copy() if client else {},
                                "holds": {},
                                "state": "pose_not_clear",
                                "last_conf": round(partial_conf, 2),
                                "visibility_score": round(pose_vis_score, 2),
                                "full_body_visible": True,
                                "ready_to_start": False
                            })
                        else:
                            # ✅ เห็นร่างกายเต็มตัวและจุดสำคัญครบ -> เริ่มตรวจจับและนับ
                            confidence = analyzer.detect(selected_pose, landmarks)

                            # ✅ CRITICAL: อัพเดท counters (จะนับก็ต่อเมื่อเห็นเต็มตัว)
                            clients.update_counters(client_id, selected_pose, confidence, ts, full_body_visible)
                            
                            # ✅ ดึงข้อมูลล่าสุดหลังจาก update
                            current_reps = client.reps_counts.copy() if client else {}
                            current_holds = {}
                            
                            if selected_pose in ["Plank", "Side Plank"]:
                                hold_data = clients.get_hold_time(client_id, selected_pose)
                                current_holds = {
                                    selected_pose: {
                                        "current_hold": hold_data["current"],
                                        "best_hold": hold_data["best"]
                                    }
                                }
                            
                            hold_time = current_holds.get(selected_pose, {}).get("current_hold", 0.0)
                            advice_msg = analyzer.feedback(selected_pose, landmarks, confidence, hold_time)

                            # ✅ อัพเดท response ด้วยข้อมูลล่าสุด
                            response.update({
                                "confidence": round(float(confidence), 3),
                                "advice": advice_msg,
                                "reps": current_reps,  # ✅ ส่งค่าล่าสุด
                                "holds": current_holds,
                                "state": clients.get_state_debug(client_id, selected_pose),
                                "last_conf": round(confidence, 2),
                                "visibility_score": round(pose_vis_score, 2),
                                "full_body_visible": True,
                                "ready_to_start": True
                            })
                            
                            # ✅ DEBUG LOG
                            logger.info(f"[{client_id}] {selected_pose} - Conf: {confidence:.2f}, Reps: {current_reps.get(selected_pose, 0)}, State: {clients.get_state_debug(client_id, selected_pose)}")
                else:
                    # ไม่เจอ landmarks เลย
                    response.update({
                        "confidence": 0.0,
                        "advice": "กรุณาเข้ามาในกรอบกล้อง",
                        "reps": client.reps_counts.copy() if client else {},
                        "holds": {},
                        "state": "no_person_detected",
                        "last_conf": 0.0,
                        "visibility_score": 0.0,
                        "full_body_visible": False,
                        "ready_to_start": False
                    })

            # ✅ ส่ง response กลับไป
            await websocket.send_json(response)

    except WebSocketDisconnect:
        clients.remove(client_id)
        logger.info(f"[DISCONNECTED] {client_id}")
    except Exception as e:
        logger.error(f"[UNEXPECTED ERROR] {e}", exc_info=True)
        clients.remove(client_id)

# ---------------- HTTP ----------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Fitness Pose Detection API",
        "version": "3.2.0",
        "description": "Real-time exercise pose detection and tracking for fitness applications",
        "available_poses": list(PoseAnalyzer.DETECTORS.keys()),
        "active_clients": clients.count(),
        "features": [
            "Full body visibility detection (must see entire body to start counting)",
            "Rep counting for 7 exercises (Squat, Push-ups, Sit-ups, etc.)",
            "Hold time tracking for 2 exercises (Plank, Side Plank)",
            "Real-time form feedback in Thai",
            "Confidence scoring (0-20% when body not visible, 0-100% when visible)",
            "Pose-specific landmark validation"
        ],
        "websocket_endpoint": "/ws/pose",
        "http_endpoints": {
            "root": "/",
            "health": "/health",
            "poses": "/poses"
        },
        "documentation": "See API docs for integration details"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "active_clients": clients.count(), "timestamp": time.time()}

@app.get("/poses")
async def list_poses():
    """รายการท่าออกกำลังกายทั้งหมด"""
    return {
        "hold_poses": {
            "description": "Time-based exercises (count seconds held)",
            "exercises": list(PoseAnalyzer.HOLD_POSES),
            "note": "Timer starts only when full body is visible and form is correct"
        },
        "rep_poses": {
            "description": "Repetition-based exercises (count reps)",
            "exercises": list(PoseAnalyzer.REPS_POSES),
            "note": "Reps counted only when full body is visible throughout movement"
        },
        "all_poses": list(PoseAnalyzer.DETECTORS.keys()),
        "tracking_requirements": {
            "full_body_visible": "Must see shoulders, hips, knees, and ankles",
            "min_visibility": "At least 6 out of 8 key landmarks visible",
            "confidence_range": "0-20% when body not visible, 0-100% when visible"
        }
    }

@app.get("/debug/client/{client_id}")
async def debug_client(client_id: str):
    """ดูข้อมูล debug ของ client เฉพาะ"""
    client = clients.clients.get(client_id)
    if not client:
        return {"error": "Client not found"}
    
    return {
        "client_id": client_id,
        "selected_pose": client.selected_pose,
        "reps_counts": client.reps_counts,
        "hold_times": client.hold_times,
        "pose_states": client.pose_states,
        "last_confidence": client.last_confidence,
        "peak_detected": client.peak_detected,
        "confidence_history": client.confidence_history,
        "thresholds": {
            "HIGH_THRESHOLD": ClientManager.HIGH_THRESHOLD,
            "LOW_THRESHOLD": ClientManager.LOW_THRESHOLD,
            "TRANSITION_THRESHOLD": ClientManager.TRANSITION_THRESHOLD,
            "HOLD_THRESHOLD": ClientManager.HOLD_THRESHOLD
        }
    }

@app.get("/debug/all")
async def debug_all():
    """ดูข้อมูล debug ของ clients ทั้งหมด"""
    return {
        "active_clients": clients.count(),
        "clients": {
            cid: {
                "pose": c.selected_pose,
                "reps": c.reps_counts,
                "holds": c.hold_times,
                "state": c.pose_states.get(c.selected_pose, "N/A") if c.selected_pose else "N/A",
                "last_conf": c.last_confidence.get(c.selected_pose, 0.0) if c.selected_pose else 0.0
            }
            for cid, c in clients.clients.items()
        },
        "thresholds": {
            "HIGH_THRESHOLD": ClientManager.HIGH_THRESHOLD,
            "LOW_THRESHOLD": ClientManager.LOW_THRESHOLD,
            "TRANSITION_THRESHOLD": ClientManager.TRANSITION_THRESHOLD,
            "HOLD_THRESHOLD": ClientManager.HOLD_THRESHOLD
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 60)
    logger.info("Fitness Pose Detection API v3.2 (DEBUG)")
    logger.info("=" * 60)
    logger.info("WebSocket: ws://0.0.0.0:8000/ws/pose")
    logger.info("HTTP API: http://0.0.0.0:8000")
    logger.info("=" * 60)
    logger.info("Full body visibility required to start counting")
    logger.info("Hold poses: Plank, Side Plank (time-based)")
    logger.info("Rep poses: Squat, Push-ups, Sit-ups, etc. (count-based)")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")