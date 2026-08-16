import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import gc
import math
import time
import json
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import cv2

app = FastAPI(title="LogiNexus Master Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

DB_PATH = "potholes.db"
yolo_model = None
INFERENCE_BUSY = False

COCO_REJECT_CLASSES = {
    'person', 'face', 'cell phone', 'chair', 'bottle', 'cup', 'laptop',
    'mouse', 'keyboard', 'tv', 'book', 'couch', 'bed', 'dining table',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'car', 'bus', 'truck'
}

SYSTEM_SETTINGS = {
    "consensus_radius_m": 15.0,
    "vibration_threshold_g": 1.7,
    "yolo_direct_threshold": 0.65,
    "yolo_min_threshold": 0.30,
    "telemetry_interval_ms": 300,
    "default_map_layer": "dark"
}

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn

def init_db():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            pin TEXT,
            role TEXT,
            full_name TEXT,
            vehicle_info TEXT,
            token TEXT
        )
        """)

        default_users = [
            ("USR-D1", "driver1", "1234", "driver", "Adarsh (Driver 01)", "Bus MH-31-A-101", "TOKEN-D1-2026"),
            ("USR-D2", "driver2", "1234", "driver", "Rajesh (Driver 02)", "Patrol Car MH-31-B-202", "TOKEN-D2-2026"),
            ("USR-D3", "driver3", "1234", "driver", "Amit (Driver 03)", "Transit Van MH-31-C-303", "TOKEN-D3-2026"),
            ("USR-NMC", "nmc_admin", "1234", "municipal", "Er. R. Sharma", "Executive Municipal Engineer", "TOKEN-NMC-OFFICER-2026")
        ]
        cur.executemany("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?, ?, ?)", default_users)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS defects (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            created_epoch REAL,
            date_str TEXT,
            day_offset INTEGER,
            latitude REAL,
            longitude REAL,
            speed REAL,
            confidence REAL,
            box_w REAL,
            box_h REAL,
            severity TEXT,
            est_diameter_cm REAL,
            asphalt_quota REAL,
            vibration_class TEXT,
            image_path TEXT,
            consensus_count INTEGER,
            drivers_list TEXT,
            status TEXT,
            location_name TEXT,
            notes TEXT,
            driver_id TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id TEXT PRIMARY KEY,
            ticket_id TEXT,
            timestamp TEXT,
            created_epoch REAL,
            latitude REAL,
            longitude REAL,
            speed REAL,
            confidence REAL,
            severity TEXT,
            trigger_source TEXT,
            image_path TEXT,
            driver_id TEXT,
            consensus_count INTEGER
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            val TEXT
        )
        """)
        for k, v in SYSTEM_SETTINGS.items():
            cur.execute("INSERT OR IGNORE INTO system_settings VALUES (?, ?)", (k, str(v)))

        cur.execute("PRAGMA table_info(detection_logs)")
        dl_cols = [c[1] for c in cur.fetchall()]
        if 'speed' not in dl_cols and len(dl_cols) > 0:
            cur.execute("ALTER TABLE detection_logs ADD COLUMN speed REAL DEFAULT 0.0")

        cur.execute("PRAGMA table_info(defects)")
        def_cols = [c[1] for c in cur.fetchall()]
        if 'speed' not in def_cols and len(def_cols) > 0:
            cur.execute("ALTER TABLE defects ADD COLUMN speed REAL DEFAULT 0.0")

        conn.commit()
    finally:
        conn.close()

init_db()

try:
    import torch
    torch.set_num_threads(1)
    from ultralytics import YOLO
    model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
    yolo_model = YOLO(model_path, task="detect")
    print(f"[ML ENGINE] Low-RAM Engine Initialized: {model_path}")
except Exception as e:
    print(f"[ERROR] ML load failed: {e}")

class DriverSession:
    def __init__(self, driver_id: str):
        self.driver_id = driver_id
        self.last_update = time.time()
        self.last_lat: Optional[float] = None
        self.last_lon: Optional[float] = None
        self.total_distance_km = 0.0

    def update_position(self, lat: float, lon: float):
        now = time.time()
        if self.last_lat is not None and self.last_lon is not None:
            delta = haversine_distance(self.last_lat, self.last_lon, lat, lon)
            time_delta_h = (now - self.last_update) / 3600.0
            if time_delta_h > 0 and (delta / time_delta_h) < 160:
                self.total_distance_km += delta
        self.last_lat = lat
        self.last_lon = lon
        self.last_update = now

drivers_db: Dict[str, DriverSession] = {}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    a = max(0.0, min(1.0, a))
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))

def is_human_skin_or_face(crop_bgr: np.ndarray) -> bool:
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / (crop_bgr.shape[0] * crop_bgr.shape[1] + 1e-5)
    return skin_ratio > 0.40

def classify_vibration(ax: float, ay: float, az: float, speed: float, threshold_g: float) -> dict:
    vertical_spike_ms2 = abs(az - 9.81)
    vertical_spike_g = vertical_spike_ms2 / 9.81
    lateral_force_ms2 = math.sqrt(ax**2 + ay**2)
    axis_ratio = lateral_force_ms2 / (vertical_spike_ms2 + 1e-4)

    if (speed < 3.0 and vertical_spike_g > 1.8) or (axis_ratio > 1.30 and vertical_spike_g > 1.2):
        vibe_class = "PHONE_DROP"
        is_pothole_shock = False
        conf = 0.0
    elif az > 13.0 and vertical_spike_g < 2.4 and axis_ratio < 0.65:
        vibe_class = "SPEED_BREAKER"
        is_pothole_shock = False
        conf = 0.2
    elif vertical_spike_g >= threshold_g and axis_ratio < 0.85:
        vibe_class = "POTHOLE_IMPACT"
        is_pothole_shock = True
        conf = min(1.0, round(vertical_spike_g / 3.5, 2))
    elif vertical_spike_g > 0.9:
        vibe_class = "ROUGH_ROAD"
        is_pothole_shock = False
        conf = 0.3
    else:
        vibe_class = "SMOOTH_ROAD"
        is_pothole_shock = False
        conf = 0.0

    return {
        "vibration_class": vibe_class,
        "is_anomaly": is_pothole_shock,
        "vertical_spike_g": round(vertical_spike_g, 2),
        "lateral_force_ms2": round(lateral_force_ms2, 2),
        "confidence": round(conf, 2)
    }

def run_model_inference(img_bytes: bytes, min_conf: float):
    global INFERENCE_BUSY
    if INFERENCE_BUSY:
        return [], None
    
    INFERENCE_BUSY = True
    detections = []
    image = None
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image.thumbnail((320, 320))
        np_img = np.array(image)
        np_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        frame_h, frame_w, _ = np_bgr.shape
        
        if yolo_model:
            results = yolo_model.predict(source=image, conf=min_conf, imgsz=320, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = str(yolo_model.names.get(cls_id, cls_id)).lower()
                    conf = float(box.conf[0])
                    
                    if cls_name in COCO_REJECT_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bw = x2 - x1
                    bh = y2 - y1

                    if bh > (bw * 1.35) or y1 < (frame_h * 0.15):
                        continue

                    crop = np_bgr[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                    if is_human_skin_or_face(crop):
                        continue

                    if conf >= min_conf:
                        detections.append({
                            "x": x1, "y": y1,
                            "w": bw, "h": bh,
                            "confidence": round(conf, 2)
                        })
    finally:
        INFERENCE_BUSY = False
        gc.collect()

    return detections, image

@app.get("/")
def serve_home():
    return FileResponse("index.html")

@app.post("/api/v1/auth/login")
def login(username: str = Form(...), pin: str = Form(...)):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, username, role, full_name, vehicle_info, token FROM users WHERE username = ? AND pin = ?", (username.strip().lower(), pin.strip()))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid username or PIN.")
        return {
            "status": "SUCCESS",
            "user": {
                "id": row[0], "username": row[1], "role": row[2],
                "full_name": row[3], "vehicle_info": row[4], "token": row[5]
            }
        }
    finally:
        conn.close()

@app.get("/api/v1/settings")
def get_settings():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, val FROM system_settings")
        rows = cur.fetchall()
        return {r[0]: (float(r[1]) if r[1].replace('.','',1).isdigit() else r[1]) for r in rows}
    finally:
        conn.close()

@app.post("/api/v1/settings")
def update_settings(
    consensus_radius_m: float = Form(15.0),
    vibration_threshold_g: float = Form(1.7),
    yolo_direct_threshold: float = Form(0.65),
    telemetry_interval_ms: int = Form(300),
    default_map_layer: str = Form("dark")
):
    SYSTEM_SETTINGS.update({
        "consensus_radius_m": consensus_radius_m,
        "vibration_threshold_g": vibration_threshold_g,
        "yolo_direct_threshold": yolo_direct_threshold,
        "telemetry_interval_ms": telemetry_interval_ms,
        "default_map_layer": default_map_layer
    })

    conn = get_db()
    try:
        cur = conn.cursor()
        for k, v in SYSTEM_SETTINGS.items():
            cur.execute("REPLACE INTO system_settings VALUES (?, ?)", (k, str(v)))
        conn.commit()
        return {"status": "SUCCESS", "settings": SYSTEM_SETTINGS}
    finally:
        conn.close()

@app.post("/api/v1/database/clear")
def clear_database():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM defects")
        cur.execute("DELETE FROM detection_logs")
        conn.commit()
        return {"status": "SUCCESS", "message": "All database records wiped."}
    finally:
        conn.close()

@app.post("/api/v1/defects/delete")
def delete_defect_log(ticket_id: str = Form(...)):
    conn = get_db()
    try:
        cur = conn.cursor()
        actual_ticket_id = ticket_id
        if ticket_id.startswith("LOG-"):
            cur.execute("SELECT ticket_id, image_path FROM detection_logs WHERE id = ?", (ticket_id,))
            row = cur.fetchone()
            if row:
                actual_ticket_id = row[0]
                if row[1] and row[1].startswith("/uploads/"):
                    dp = row[1].lstrip("/")
                    if os.path.exists(dp):
                        try: os.remove(dp)
                        except: pass
        
        cur.execute("SELECT image_path FROM defects WHERE id = ?", (actual_ticket_id,))
        d_row = cur.fetchone()
        if d_row and d_row[0] and d_row[0].startswith("/uploads/"):
            dp = d_row[0].lstrip("/")
            if os.path.exists(dp):
                try: os.remove(dp)
                except: pass

        cur.execute("DELETE FROM defects WHERE id = ? OR id = ?", (actual_ticket_id, ticket_id))
        cur.execute("DELETE FROM detection_logs WHERE ticket_id = ? OR id = ?", (actual_ticket_id, ticket_id))
        conn.commit()
        return {"status": "SUCCESS", "deleted_ticket": actual_ticket_id}
    finally:
        conn.close()

@app.post("/api/v1/logs/delete-batch")
def delete_batch_logs(log_ids: str = Form(...)):
    conn = get_db()
    try:
        cur = conn.cursor()
        id_list = [i.strip() for i in log_ids.split(",") if i.strip()]
        if not id_list:
            return {"status": "SUCCESS", "message": "No IDs provided."}
        
        placeholders = ",".join(["?"] * len(id_list))
        cur.execute(f"SELECT ticket_id, image_path FROM detection_logs WHERE id IN ({placeholders})", id_list)
        rows = cur.fetchall()
        
        ticket_ids_to_check = set()
        for r in rows:
            if r[0]: ticket_ids_to_check.add(r[0])
            if r[1] and r[1].startswith("/uploads/"):
                dp = r[1].lstrip("/")
                if os.path.exists(dp):
                    try: os.remove(dp)
                    except: pass

        cur.execute(f"DELETE FROM detection_logs WHERE id IN ({placeholders})", id_list)
        if ticket_ids_to_check:
            t_placeholders = ",".join(["?"] * len(ticket_ids_to_check))
            cur.execute(f"DELETE FROM defects WHERE id IN ({t_placeholders})", list(ticket_ids_to_check))
        cur.execute(f"DELETE FROM defects WHERE id IN ({placeholders})", id_list)

        conn.commit()
        return {"status": "SUCCESS", "deleted_count": len(id_list)}
    finally:
        conn.close()

@app.post("/api/v1/detect")
@app.post("/api/v1/telemetry")
async def process_telemetry(
    file: UploadFile = File(...),
    latitude: float = Form(21.1458),
    longitude: float = Form(79.0882),
    speed: float = Form(0.0),
    accel_x: float = Form(0.0),
    accel_y: float = Form(0.0),
    accel_z: float = Form(9.81),
    driver_id: str = Form("driver1"),
    notes: str = Form("")
):
    t_start = time.perf_counter()
    if driver_id not in drivers_db:
        drivers_db[driver_id] = DriverSession(driver_id)
    driver = drivers_db[driver_id]
    driver.update_position(latitude, longitude)

    vibe_threshold = SYSTEM_SETTINGS.get("vibration_threshold_g", 1.7)
    vibe_eval = classify_vibration(accel_x, accel_y, accel_z, speed, vibe_threshold)

    image_bytes = await file.read()
    min_vision_conf = SYSTEM_SETTINGS.get("yolo_min_threshold", 0.30)
    direct_conf = SYSTEM_SETTINGS.get("yolo_direct_threshold", 0.65)
    
    detections, image = await asyncio.to_thread(run_model_inference, image_bytes, min_vision_conf)
    
    vision_detected = len(detections) > 0
    vision_conf = max([d["confidence"] for d in detections]) if vision_detected else 0.0

    is_pothole_confirmed = False
    confirmed_by = "NONE"
    fused_conf = 0.0

    if vision_detected and vision_conf >= direct_conf:
        is_pothole_confirmed = True
        fused_conf = vision_conf
        confirmed_by = f"DIRECT_VISION ({int(vision_conf*100)}%)"

    elif vision_detected and vision_conf >= min_vision_conf and vibe_eval["is_anomaly"]:
        is_pothole_confirmed = True
        fused_conf = round((vision_conf * 0.60) + (vibe_eval["confidence"] * 0.40), 2)
        confirmed_by = f"DUAL_SENSOR (Vis: {int(vision_conf*100)}% + Shock: {vibe_eval['vertical_spike_g']}G)"

    elif vibe_eval["is_anomaly"] and vibe_eval["vibration_class"] == "POTHOLE_IMPACT":
        is_pothole_confirmed = True
        fused_conf = max(0.75, vibe_eval["confidence"])
        confirmed_by = f"VIBRATION_IMPACT ({vibe_eval['vertical_spike_g']}G Spike)"

    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    if not is_pothole_confirmed:
        return {
            "status": "CLEAR",
            "vibration_class": vibe_eval["vibration_class"],
            "driver_metrics": {"driver_id": driver.driver_id, "distance_km": round(driver.total_distance_km, 2)},
            "latency_ms": latency_ms,
            "vibration": vibe_eval,
            "detections": detections
        }

    best_det = detections[0] if detections else {"w": 80, "h": 50, "confidence": fused_conf}
    if vibe_eval["vertical_spike_g"] >= 2.5 or fused_conf >= 0.85 or best_det["w"] >= 95:
        severity = "High"
        diameter_cm = 75
    elif vibe_eval["vertical_spike_g"] >= 1.6 or fused_conf >= 0.60 or best_det["w"] >= 45:
        severity = "Medium"
        diameter_cm = 45
    else:
        severity = "Low"
        diameter_cm = 25

    area_m2 = math.pi * ((diameter_cm / 200.0) ** 2)
    asphalt_mt = round(area_m2 * 0.07 * 2.35 * 1.15 + 0.30, 2)

    cluster_radius = SYSTEM_SETTINGS.get("consensus_radius_m", 15.0)
    now = datetime.now()
    ts_str = now.strftime("%d %b %Y, %I:%M %p")
    
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, latitude, longitude, consensus_count, drivers_list, status FROM defects WHERE status != 'RESOLVED' AND status != 'DISMISSED'")
        records = cur.fetchall()

        matched_id = None
        final_drivers = [driver_id]
        final_count = 1
        auto_shared = False

        for r in records:
            r_id, r_lat, r_lng, r_count, r_drivers_json, r_status = r
            dist_m = haversine_distance(latitude, longitude, r_lat, r_lng) * 1000.0
            if dist_m <= cluster_radius:
                matched_id = r_id
                driver_list = json.loads(r_drivers_json)
                if driver_id not in driver_list:
                    driver_list.append(driver_id)
                final_drivers = driver_list
                final_count = len(driver_list)
                
                if final_count >= 3:
                    new_status = "AUTO_DISPATCHED_TO_NMC"
                    auto_shared = True
                else:
                    new_status = r_status if "DISPATCH" in r_status else "CANDIDATE"

                cur.execute("""
                    UPDATE defects 
                    SET consensus_count = ?, drivers_list = ?, status = ?, latitude = ?, longitude = ?, speed = ?
                    WHERE id = ?
                """, (final_count, json.dumps(final_drivers), new_status, latitude, longitude, speed, r_id))
                conn.commit()
                break

        is_recent_duplicate = False
        if matched_id:
            cur.execute("""
                SELECT created_epoch FROM detection_logs 
                WHERE ticket_id = ? AND driver_id = ? 
                ORDER BY created_epoch DESC LIMIT 1
            """, (matched_id, driver_id))
            last_log = cur.fetchone()
            if last_log and (time.time() - last_log[0] < 10.0):
                is_recent_duplicate = True

        img_save_path = ""
        if not is_recent_duplicate and image is not None:
            filename = f"defect_{now.strftime('%Y%m%d_%H%M%S')}_{driver_id}.jpg"
            img_save_path = f"/uploads/{filename}"
            image.save(os.path.join(UPLOAD_DIR, filename), "JPEG", quality=65)

        if not matched_id:
            matched_id = f"LN-{int(time.time()) % 100000}"
            location_tag = f"Corridor {latitude:.5f}N, {longitude:.5f}E"
            cur.execute("""
                INSERT INTO defects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                matched_id, ts_str, time.time(), now.strftime("%Y-%m-%d"), 1,
                latitude, longitude, speed, round(fused_conf, 2),
                best_det["w"], best_det["h"], severity, diameter_cm, asphalt_mt,
                confirmed_by, img_save_path,
                1, json.dumps([driver_id]), "CANDIDATE",
                location_tag, notes, driver_id
            ))
            conn.commit()

        log_id = None
        if not is_recent_duplicate:
            log_id = f"LOG-{int(time.time() * 1000) % 1000000}"
            cur.execute("""
                INSERT INTO detection_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_id, matched_id, ts_str, time.time(),
                latitude, longitude, speed, round(fused_conf, 2),
                severity, confirmed_by, img_save_path, driver_id, final_count
            ))
            conn.commit()
    finally:
        conn.close()

    return {
        "status": "SUCCESS",
        "ticket_id": matched_id,
        "log_id": log_id,
        "is_new_log": not is_recent_duplicate,
        "severity": severity,
        "confidence": round(fused_conf, 2),
        "vibration_class": confirmed_by,
        "asphalt_quota_mt": asphalt_mt,
        "box_size": f"{best_det['w']}x{best_det['h']} px",
        "diameter_cm": diameter_cm,
        "consensus_count": final_count,
        "drivers_list": final_drivers,
        "is_verified": final_count >= 3,
        "auto_dispatched_to_municipal": auto_shared,
        "image_url": img_save_path,
        "timestamp": ts_str,
        "latency_ms": latency_ms,
        "driver_metrics": {"driver_id": driver.driver_id, "distance_km": round(driver.total_distance_km, 2)},
        "detections": detections
    }

@app.post("/api/v1/manual-report")
async def process_manual_report(
    file: UploadFile = File(...),
    latitude: float = Form(21.1458),
    longitude: float = Form(79.0882),
    driver_id: str = Form("driver1"),
    notes: str = Form(""),
    target_dept: str = Form("Municipal Corporation")
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((480, 480))
    
    detections, _ = run_model_inference(image_bytes, 0.30)
    best_det = detections[0] if detections else {"w": 140, "h": 95, "confidence": 0.85}
    severity = "High" if best_det["w"] > 70 or best_det["confidence"] > 0.70 else "Medium"
    diameter_cm = 75 if severity == "High" else 45
    asphalt_mt = round(math.pi * ((diameter_cm / 200.0) ** 2) * 0.07 * 2.35 * 1.15 + 0.30, 2)

    now = datetime.now()
    ts_str = now.strftime("%d %b %Y, %I:%M %p")
    filename = f"manual_{now.strftime('%Y%m%d_%H%M%S')}_{driver_id}.jpg"
    img_save_path = f"/uploads/{filename}"
    image.save(os.path.join(UPLOAD_DIR, filename), "JPEG", quality=70)

    ticket_id = f"MNC-{int(time.time()) % 100000}"
    location_tag = f"GPS {latitude:.5f}N, {longitude:.5f}E"

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO defects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id, ts_str, time.time(), now.strftime("%Y-%m-%d"), 1,
            latitude, longitude, 0.0, best_det["confidence"],
            best_det["w"], best_det["h"], severity, diameter_cm, asphalt_mt,
            "MANUAL_PHOTO", img_save_path,
            3, json.dumps([driver_id]), "AUTO_DISPATCHED_TO_NMC",
            location_tag, notes, driver_id
        ))
        
        log_id = f"LOG-{int(time.time() * 1000) % 1000000}"
        cur.execute("""
            INSERT INTO detection_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_id, ticket_id, ts_str, time.time(),
            latitude, longitude, 0.0, best_det["confidence"],
            severity, "MANUAL_DISPATCH", img_save_path, driver_id, 3
        ))
        conn.commit()
    finally:
        conn.close()

    return {
        "status": "SUCCESS",
        "ticket_id": ticket_id,
        "severity": severity,
        "asphalt_quota_mt": asphalt_mt,
        "image_url": img_save_path,
        "location": location_tag,
        "timestamp": ts_str,
        "message": f"Dispatched to {target_dept}!"
    }

@app.get("/api/v1/potholes")
def get_potholes(day_offset: Optional[int] = None, severity: str = "ALL", driver_id: Optional[str] = None):
    conn = get_db()
    try:
        cur = conn.cursor()
        query = "SELECT * FROM defects WHERE 1=1"
        params = []
        if severity != "ALL":
            query += " AND severity = ?"
            params.append(severity.capitalize())
        query += " ORDER BY created_epoch DESC"
        cur.execute(query, params)
        rows = cur.fetchall()
        return {
            "count": len(rows),
            "potholes": [{
                "id": r[0], "timestamp": r[1], "latitude": r[5], "longitude": r[6], "speed": r[7],
                "confidence": r[8], "box_w": r[9], "box_h": r[10], "severity": r[11],
                "diameter_cm": r[12], "asphalt_quota": r[13], "vibration_class": r[14],
                "image_url": r[15], "consensus_count": r[16], "drivers_list": json.loads(r[17]),
                "status": r[18], "location_name": r[19], "notes": r[20], "driver_id": r[21]
            } for r in rows]
        }
    finally:
        conn.close()

@app.get("/api/v1/history")
def get_history(max_days: int = 30, driver_id: Optional[str] = None):
    conn = get_db()
    try:
        cur = conn.cursor()
        query = "SELECT * FROM defects WHERE day_offset <= ? ORDER BY created_epoch DESC"
        cur.execute(query, [max_days])
        rows = cur.fetchall()
        return {
            "max_days": max_days,
            "count": len(rows),
            "history": [{
                "id": r[0], "timestamp": r[1], "date_str": r[3], "day_offset": r[4],
                "latitude": r[5], "longitude": r[6], "speed": r[7], "severity": r[11], "diameter_cm": r[12],
                "asphalt_quota": r[13], "vibration_class": r[14], "image_url": r[15],
                "consensus_count": r[16], "drivers_list": json.loads(r[17]), "status": r[18],
                "location_name": r[19], "notes": r[20], "driver_id": r[21]
            } for r in rows]
        }
    finally:
        conn.close()

@app.get("/api/v1/logs")
def get_all_detection_logs(driver_id: Optional[str] = None):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, ticket_id, timestamp, created_epoch, latitude, longitude, speed, confidence, severity, trigger_source, image_path, driver_id, consensus_count FROM detection_logs ORDER BY created_epoch DESC")
        rows = cur.fetchall()
        return {
            "count": len(rows),
            "logs": [{
                "id": r[0],
                "ticket_id": r[1],
                "timestamp": r[2],
                "latitude": r[4],
                "longitude": r[5],
                "speed": r[6],
                "confidence": r[7],
                "severity": r[8],
                "vibration_class": r[9],
                "image_url": r[10],
                "driver_id": r[11],
                "consensus_count": r[12]
            } for r in rows]
        }
    finally:
        conn.close()

@app.get("/api/v1/analytics")
def get_analytics():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT severity, vibration_class, asphalt_quota, status, date_str FROM defects")
        rows = cur.fetchall()

        counts = {"High": 0, "Medium": 0, "Low": 0}
        asphalt_by_sev = {"High": 0.0, "Medium": 0.0, "Low": 0.0}
        status_counts = {"AUTO_DISPATCHED": 0, "CREW_DISPATCHED": 0, "RESOLVED": 0, "CANDIDATE": 0}
        total_asphalt = 0.0
        active_penalties = 0

        for r in rows:
            sev, v_class, asp, stat, d_str = r
            if stat not in ["RESOLVED", "DISMISSED"]:
                if sev in counts:
                    counts[sev] += 1
                    asphalt_by_sev[sev] += asp
                total_asphalt += asp
                active_penalties += (12 if sev == "High" else (6 if sev == "Medium" else 2))

            if "DISPATCH" in stat or "AUTO" in stat:
                status_counts["AUTO_DISPATCHED"] += 1
            elif stat in ["CREW_DISPATCHED", "IN_PROGRESS"]:
                status_counts["CREW_DISPATCHED"] += 1
            elif stat in ["RESOLVED", "REPAIRED"]:
                status_counts["RESOLVED"] += 1
            else:
                status_counts["CANDIDATE"] += 1

        dynamic_score = max(10, 100 - active_penalties)

        return {
            "dynamic_road_score": dynamic_score,
            "severity_distribution": counts,
            "asphalt_by_severity": {k: round(v, 2) for k, v in asphalt_by_sev.items()},
            "status_distribution": status_counts,
            "total_asphalt_mt": round(total_asphalt, 2),
            "total_active": sum(counts.values())
        }
    finally:
        conn.close()

@app.post("/api/v1/municipal/verify")
def verify_municipal_action(
    ticket_id: str = Form(...),
    action: str = Form("DISPATCH"),
    auth_token: Optional[str] = Header(None)
):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE token = ?", (auth_token,))
        user_row = cur.fetchone()

        if not user_row or user_row[0] != "municipal":
            raise HTTPException(status_code=403, detail="Municipal Officer credentials required.")

        status_map = {
            "DISPATCH": "CREW_DISPATCHED",
            "REPAIR": "RESOLVED",
            "DISMISS": "DISMISSED"
        }
        new_status = status_map.get(action.upper(), "CREW_DISPATCHED")

        cur.execute("UPDATE defects SET status = ? WHERE id = ?", (new_status, ticket_id))
        conn.commit()
        return {"status": "SUCCESS", "message": f"Defect {ticket_id} updated to {new_status}."}
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
