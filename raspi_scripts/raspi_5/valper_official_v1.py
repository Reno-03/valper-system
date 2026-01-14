from datetime import datetime
from picamera2 import Picamera2
from insightface.app import FaceAnalysis
from supabase import create_client, Client
from dotenv import load_dotenv
from OCR_postprocessing_v3 import *
from zeroconf import Zeroconf, ServiceBrowser, ServiceListener

import threading
import queue
import time
import uuid
import re
import atexit
import pygame
import serial
import onnxruntime as ort
import easyocr
import os
import cv2
import numpy as np
import requests
import time
import socket, time

# Load .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment (.env)")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============ ZERO CONF ===============
SERVICE_TYPE = "_demo._tcp.local."

class PiZeroListener(ServiceListener):
    def __init__(self):
        self.server_url = None

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info and info.addresses:
            ip = socket.inet_ntoa(info.addresses[0])
            port = info.port
            self.server_url = f"http://{ip}:{port}/capture"
            print(f"[rpi5] Discovered PiZero at {self.server_url}")

# in capture_loop() before fetch attempts:
zeroconf = Zeroconf()
listener = PiZeroListener()
browser = ServiceBrowser(zeroconf, SERVICE_TYPE, listener)

# wait up to N seconds for discovery
timeout = time.time() + 10
while listener.server_url is None and time.time() < timeout:
    time.sleep(0.2)

if listener.server_url is None:
    print("[ERROR] PiZero service not found. Either set Pi_Zero_URL manually or increase timeout.")
    # fallback/hardcode option:
    PI_ZERO_URL = "http://10.232.140.155:5001/capture"
else:
    Pi_Zero_URL = listener.server_url
    print(f"[rpi5] Using PiZero URL: {Pi_Zero_URL}")



# ========== CONFIG ==========
UPLOAD_FOLDER = './captured_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


MODEL_PATH = r"/home/user/my_pi_camera_project/VALPER Python/best.onnx"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6


# Global to keep last load faces timing (happens on startup)
last_load_faces_time = None


# ========== Model initialization ==========
print("[DEBUG] Initializing FaceAnalysis...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))


print("[DEBUG] Initializing OCR...")
ocr_reader = easyocr.Reader(['en'], gpu=False)


try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    print(f"[DEBUG] Loading ONNX model: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Could not load ONNX model: {e}")
    exit(1)


# ========== Audio Setup ==========
# pygame.mixer.init()
AUDIO_PATH = "VALPER Python/tts-VALPER-Official/"

# Initialize mixer once
pygame.mixer.init()

# Queue to hold audio files
audio_queue = queue.Queue()

# Background thread to process audio queue
def audio_player_thread():
    while True:
        filename = audio_queue.get()  # wait for next audio
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            # Wait until this audio finishes before playing the next
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Audio error: {e}")
        audio_queue.task_done()

# Start the audio thread (daemon so it exits with main program)
threading.Thread(target=audio_player_thread, daemon=True).start()

# Helper function to enqueue audio
def play_audio(filename):
    audio_queue.put(AUDIO_PATH + filename)

# ========== Serial Setup ==========
try:
    print("[DEBUG] Connecting to Arduino...")
    ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)
    time.sleep(2)
    print("[DEBUG] Serial connection established")
except Exception as e:
    print(f"[ERROR] Could not connect to serial: {e}")
    exit()

# ========== FACE DATABASE ==========
def load_faces_from_supabase():
    global last_load_faces_time
    print("[DEBUG] Loading face embeddings from Supabase...")
    start = time.time()
    try:
        response = supabase.table("face_embeddings").select("*").execute()
        face_db = {}
        if not response.data:
            print("[WARNING] No face embeddings found in Supabase.")
            last_load_faces_time = time.time() - start
            print(f"[PERF] Step run load face embeddings from Supabase took {last_load_faces_time:.4f} sec")
            return face_db, last_load_faces_time
        for row in response.data:
            name = row["name"]
            emb = np.array(row["embedding"], dtype=np.float32).ravel()  # force 1D
            if name not in face_db:
                face_db[name] = []
            face_db[name].append(emb)
        loaded_count = sum(len(v) for v in face_db.values())
        elapsed = time.time() - start
        last_load_faces_time = elapsed
        print(f"[INFO] Loaded {loaded_count} embeddings from Supabase")
        print(f"[PERF] Step run load face embeddings from Supabase took {elapsed:.4f} sec")
        return face_db, elapsed
    except Exception as e:
        print(f"[ERROR] Failed to load face data: {e}")
        last_load_faces_time = time.time() - start
        print(f"[PERF] Step run load face embeddings from Supabase took {last_load_faces_time:.4f} sec")
        return {}, last_load_faces_time

known_faces, last_load_faces_time = load_faces_from_supabase()


# ========== CAMERA ==========
picam2 = None

def init_camera_preview():
    global picam2
    print("[DEBUG] Initializing camera in PREVIEW mode...")
    if picam2:
        try:
            picam2.stop()
        except Exception:
            pass
    if picam2 is None:
        picam2 = Picamera2()
    preview_cfg = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
    picam2.configure(preview_cfg)
    # picam2.set_controls({"AfMode": 2})
    picam2.start()

def switch_to_still_mode():
    global picam2
    print("[DEBUG] Switching camera to STILL mode...")
    picam2.stop()
    still_cfg = picam2.create_still_configuration(main={"format": "RGB888", "size": (1920, 1080)})
    picam2.configure(still_cfg)
    # picam2.set_controls({"AfMode": 1})
    picam2.start()

def cleanup():
    global picam2
    if picam2:
        print("[DEBUG] Stopping camera...")
        picam2.stop()

atexit.register(cleanup)


# ========== HELPERS ==========
def send_to_supabase(plate, face, authorized, timings=None, direction="IN", slot_number=None):
    """
    Always logs detection to vehicle_access_raw_logs.
    If authorized, also links that raw_log to vehicle_access_logs.
    """
    print(f"[DEBUG] Logging detection: Plate={plate}, Face={face}, "
          f"Authorized={authorized}")

    upload_start = time.time()
    raw_log_id = None

    try:
        # ✅ Insert into raw logs first (always)
        raw_insert = supabase.table("vehicle_access_raw_logs").insert({
            "detected_license_plate": plate or "",
            "detected_name": face or "",
        }).execute()

        if raw_insert.data and len(raw_insert.data) > 0:
            raw_log_id = raw_insert.data[0]["id"]
            print(f"[INFO] Logged raw detection (raw_log_id={raw_log_id})")
        else:
            print("[WARN] No data returned from raw log insert.")
    except Exception as e:
        print(f"[ERROR] Failed to insert into vehicle_access_raw_logs: {e}")

    # ✅ If authorized, add to vehicle_access_logs
    if authorized and raw_log_id:
        try:
            supabase.table("vehicle_access_logs").insert({
                "license_plate": plate or "",
                "name": face or "",
                "direction": direction,
                "raw_log_id": raw_log_id
            }).execute()
            print("[INFO] Authorized — added to vehicle_access_logs.")
        except Exception as e:
            print(f"[ERROR] Failed to insert vehicle_access_logs entry: {e}")

    upload_elapsed = time.time() - upload_start
    print(f"[PERF] Step upload to Supabase took {upload_elapsed:.4f} sec")

    # --- Performance tracking ---
    if timings is None:
        timings = {}
    timings["step_upload_supabase"] = round(upload_elapsed, 4)

    run_row = {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat(),
        "step_load_face_embeddings": timings.get("step_load_face_embeddings"),
        "step_yolov8_onnx": timings.get("step_yolov8_onnx"),
        "step_easyocr": timings.get("step_easyocr"),
        "step_capture_rpi_cam3": timings.get("step_capture_rpi_cam3"),
        "step_arcface_model": timings.get("step_arcface_model"),
        "step_arcface_comparison": timings.get("step_arcface_comparison"),
        "step_upload_supabase": timings.get("step_upload_supabase"),
    }

    print(run_row)

    try:
        supabase.table("performance_runs").insert(run_row).execute()
        print("[INFO] Inserted performance_runs record.")
    except Exception as e:
        print(f"[ERROR] Failed to insert performance_runs record: {e}")

def fetch_plate_from_pi_zero():
    global Pi_Zero_URL

    print("[DEBUG] Fetching plate image from Pi Zero...")
    try:
        response = requests.get(Pi_Zero_URL, timeout=5)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            print(f"[ERROR] Bad response from Pi Zero: {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception fetching image: {e}")
        return None

def cosine_similarity(a, b):
    a = np.ravel(a).astype(np.float32)
    b = np.ravel(b).astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def resize(img, w=640, h=360):
    return cv2.resize(img, (w, h))


# ========== IMAGE PREPROCESS HELPERS ==========
def letterbox(img, new_shape=640, color=(114,114,114)):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = img_resized
    return canvas, scale, left, top

def preprocess(img):
    img, scale, left, top = letterbox(img, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img, scale, left, top

# good for most plate v 1 
def preprocess_for_ocr(plate_crop):
    """Enhance license plate image before OCR."""
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=15)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    h, w = sharp.shape
    if h < 100 or w < 200:
        sharp = cv2.resize(sharp, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return sharp

def show_ocr_preprocessing_steps(plate_crop):
    """Visualize each OCR preprocessing step."""
    original = plate_crop.copy()
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    bright = cv2.convertScaleAbs(contrast, alpha=1.4, beta=-20)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(bright, -1, kernel)

    steps = [
        ("Original", cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)),
        ("Grayscale", gray),
        ("Denoised", denoised),
        ("CLAHE", contrast),
        ("Bright/Contrast", bright),
        ("Sharpened", sharp),
    ]

    resized = [cv2.resize(img, (300, 150)) for _, img in steps]
    row1 = np.hstack(resized[:3])
    row2 = np.hstack(resized[3:])
    grid = np.vstack([row1, row2])

    annotated = grid.copy()
    y, x_step = 20, 300
    for i, (name, _) in enumerate(steps):
        row = i // 3
        col = i % 3
        x = col * x_step + 10
        y_text = row * 150 + 25
        cv2.putText(annotated, name, (x, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("OCR Preprocessing Steps", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ========== FACE RECOGNITION ==========
def perform_face_recognition(img, face_db=known_faces, threshold=0.6):
    """
    returns: (annotated_img, best_name, arcface_model_time, arcface_compare_time)
    """
    print("[DEBUG] Performing Face Recognition...")
    if img is None or not face_db:
        return img, None, None, None


    start_model = time.time()
    faces = face_app.get(img)
    arcface_model_elapsed = time.time() - start_model
    print(f"[PERF] Step run ArcFace buffalo_l Model took {arcface_model_elapsed:.4f} sec")


    if not faces:
        print("[DEBUG] No face detected.")
        return img, None, arcface_model_elapsed, None


    best_name, best_score = None, -1
    start_compare = time.time()


    for face in faces:
        for name, embs in face_db.items():
            for emb in embs:
                sim = cosine_similarity(face.embedding, emb)
                if sim > best_score:
                    best_name, best_score = name, sim
        if best_score >= threshold:
            box = face.bbox.astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(img, f"{best_name} ({best_score:.2f})",
                        (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)
            print(f"[INFO] Recognized Face: {best_name} ({best_score:.2f})")
            arcface_compare_elapsed = time.time() - start_compare
            print(f"[PERF] Step ArcFace Embedding Comparison took {arcface_compare_elapsed:.4f} sec")
            return img, best_name, arcface_model_elapsed, arcface_compare_elapsed


    # if no recognized
    arcface_compare_elapsed = time.time() - start_compare
    print("[DEBUG] No recognized face above threshold")
    return img, None, arcface_model_elapsed, arcface_compare_elapsed

def validate(plate, face):
    print(f"[DEBUG] Validating Plate={plate}, Face={face} with Supabase...")
    try:
        response = supabase.table("plate_numbers_with_users") \
            .select("*") \
            .eq("plate_number", plate) \
            .eq("full_name", face) \
            .execute()
        
        
        if response.data and len(response.data) > 0:
            print("[INFO] Validation PASSED (Supabase)")
            return True, "✅ VALID (Supabase)"
        print("[INFO] Validation FAILED (Not in Supabase)")
        return False, "❌ INVALID (Not in Supabase)"
    except Exception as e:
        print(f"[ERROR] Supabase validation failed: {e}")
        return False, "❌ ERROR (Validation Failed)"


# =====================================================
# ================ RETRY HELPER FUNCTIONS =============
# =====================================================

# -------------------------
# Improved retry helper
# -------------------------
def retry_operation(operation, max_retries=3, delay=2, description="Operation"):
    """
    Retries operation() up to max_retries times.
    operation() must return a tuple where the *key result* is in index 1:
      - LPR: (annotated_img, plate_text, yolov8_time, easyocr_time)
      - Face: (annotated_img, face_name, model_time, compare_time)
    Returns the first successful result (key result non-empty), or last_result if all fail.
    """
    last_result = None
    for attempt in range(1, max_retries + 1):
        print(f"[RETRY] {description} attempt {attempt}/{max_retries}...")
        result = operation()
        last_result = result

        # LPR: success only if plate_text (result[1]) is non-empty
        if description == "License Plate Recognition":
            if isinstance(result, tuple) and len(result) > 1:
                plate_text = result[1]
                if plate_text and str(plate_text).strip():
                    print(f"[RETRY] {description} succeeded on attempt {attempt}")
                    return result

        # Face: success only if face_name (result[1]) is non-empty
        elif description == "Face Recognition":
            if isinstance(result, tuple) and len(result) > 1:
                face_name = result[1]
                if face_name and str(face_name).strip():
                    print(f"[RETRY] {description} succeeded on attempt {attempt}")
                    return result

        # Generic fallback
        elif result is not None:
            print(f"[RETRY] {description} succeeded on attempt {attempt}")
            return result

        # failed this attempt:
        if attempt < max_retries:
            print(f"[RETRY] {description} failed on attempt {attempt}, retrying in {delay}s...")
            time.sleep(delay)
        else:
            print(f"[RETRY] {description} failed after {max_retries} attempts.")

    return last_result

# -------------------------
# LPR with refetch per retry
# -------------------------
def perform_lpr_with_retry(initial_plate_img=None, max_retries=3, refetch_each_try=True, delay_between_attempts=4):
    """
    If refetch_each_try=True, this will call fetch_plate_from_pi_zero() each attempt
    so you get fresh frames. If initial_plate_img is provided and refetch_each_try=False,
    it will retry on the same image (not recommended).
    """
    def op_factory(attempt):
        def op():
            # fetch fresh frame if requested
            img = None
            if refetch_each_try:
                img = fetch_plate_from_pi_zero()
                if img is None:
                    print(f"[DEBUG] Attempt {attempt}: failed to fetch new image from Pi Zero.")
                    return (None, None, None, None)
            else:
                img = initial_plate_img
            # call perform_lpr but pass attempt for better debug logs
            return perform_lpr(img, debug_attempt=attempt, interactive=False)
        return op

    last = None
    for i in range(1, max_retries + 1):
        result = op_factory(i)()
        last = result
        # use the same success rules as retry_operation
        if isinstance(result, tuple) and len(result) > 1 and result[1]:
            print(f"[RETRY] License Plate Recognition succeeded on attempt {i}")
            return result
        if i < max_retries:
            play_audio("Okay__No_Plate_Detected__try_again.mp3")
            print(f"[RETRY] License Plate Recognition attempt {i} failed; sleeping {delay_between_attempts}s before retry.")
            time.sleep(delay_between_attempts)

    print("[ERROR] Plate could not be detected after retries.")
    return last

def perform_lpr(img, debug_attempt=None, interactive=False):
    """
    Returns: (annotated_img, validated_text, yolov8_time, easyocr_time)
    Accepts:
      - debug_attempt: int or None -> prints attempt info for logs
      - interactive: if True, calls show_ocr_preprocessing_steps(crop)
    """
    if img is None:
        print("[DEBUG] perform_lpr called with img=None")
        return None, None, None, None

    if debug_attempt:
        print(f"[DEBUG] Performing License Plate Recognition (attempt {debug_attempt})...")
    else:
        print("[DEBUG] Performing License Plate Recognition...")

    h0, w0 = img.shape[:2]
    inp, scale, left, top = preprocess(img)

    # --- YOLOv8 Detection ---
    start = time.time()
    preds = session.run(None, {input_name: inp})[0]
    yolov8_elapsed = time.time() - start
    print(f"[PERF] YOLOv8 inference took {yolov8_elapsed:.4f}s")

    preds = np.squeeze(preds)
    print(f"[DEBUG] raw preds shape: {preds.shape}")

    if preds.ndim == 1:
        print("[DEBUG] preds seems 1D -> no detections")
        return img, None, yolov8_elapsed, None

    if preds.shape[0] == 5 and preds.shape[1] != 5:
        preds = preds.transpose(1, 0)
    if preds.shape[1] != 5:
        print(f"[ERROR] Unexpected ONNX output shape: {preds.shape}")
        return img, None, yolov8_elapsed, None

    # --- Extract Boxes ---
    boxes, scores = [], []
    for det_idx, det in enumerate(preds):
        cx, cy, w, h, conf = det
        if conf < CONF_THRESHOLD:
            continue
        x1 = int(((cx - w / 2) * INPUT_SIZE - left) / scale)
        y1 = int(((cy - h / 2) * INPUT_SIZE - top) / scale)
        x2 = int(((cx + w / 2) * INPUT_SIZE - left) / scale)
        y2 = int(((cy + h / 2) * INPUT_SIZE - top) / scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0 - 1, x2), min(h0 - 1, y2)
        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))

    print(f"[DEBUG] Filtered boxes count: {len(boxes)}; scores: {scores}")

    if not boxes:
        print("[DEBUG] No detections found.")
        return img, None, yolov8_elapsed, None

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    print(f"[DEBUG] Raw NMSBoxes indices: {indices}")

    if indices is None or len(indices) == 0:
        print("[DEBUG] No boxes after NMS.")
        return img, None, yolov8_elapsed, None

    # Normalize indices
    if hasattr(indices, "flatten"):
        try:
            indices = indices.flatten().tolist()
        except Exception:
            indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]

    print(f"[DEBUG] Final detection indices: {indices}")

    plate_text = None
    easyocr_elapsed = None

    # --- Fetch database once ---
    print("[INFO] Fetching plate database for validation...")
    database = get_database_plates()

    # --- Process each detection ---
    for i in indices:
        x1, y1, x2, y2 = boxes[i]

        # tighten crop
        w, h = x2 - x1, y2 - y1
        margin_x, margin_y = int(w * 0.1), int(h * 0.1)
        x1_ = max(0, x1 + margin_x)
        x2_ = min(w0 - 1, x2 - margin_x)
        y1_ = max(0, y1 + margin_y)
        y2_ = min(h0 - 1, y2 - margin_y)

        crop = img[y1_:y2_, x1_:x2_]
        if crop.size == 0:
            continue

        if interactive:
            show_ocr_preprocessing_steps(crop)

        enhanced_crop = preprocess_for_ocr(crop)
        start_ocr = time.time()
        detections = ocr_reader.readtext(enhanced_crop)
        easyocr_elapsed = time.time() - start_ocr
        print(f"[PERF] EasyOCR took {easyocr_elapsed:.4f}s; OCR raw: {detections}")

        if not detections:
            print("[DEBUG] OCR returned no text for this crop, trying next candidate.")
            continue

        # --- Clean OCR output ---
        ocr_text = re.sub(r'[^A-Z0-9]', '', detections[0][1].upper())
        print(f"[DEBUG] OCR cleaned text: {ocr_text}")

        # --- 🔬 Research-Backed Validation ---
        print("[INFO] Running advanced OCR validation (ensemble method)...")
        result = validate_plate_research_backed(ocr_text, database)
        display_result_advanced(result)

        if result.get("validated"):
            plate_text = result["plate"]
            cv2.rectangle(img, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1_, y1_ - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"[✅ INFO] Plate validated: {plate_text} ({result['confidence']}, {result['reliability']:.1f}%)")
        else:
            plate_text = ocr_text
            cv2.rectangle(img, (x1_, y1_), (x2_, y2_), (0, 255, 255), 2)
            cv2.putText(img, plate_text, (x1_, y1_ - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"[⚠️ INFO] OCR text (unvalidated): {plate_text}")

        # Stop after first detection for simplicity
        break

    return img, plate_text, yolov8_elapsed, easyocr_elapsed

def perform_face_recognition_with_retry(max_retries=3, delay_between_attempts=4, preview_duration=3):
    """
    Perform Face Recognition with retry mechanism.
    Shows camera preview before each capture attempt, then performs recognition.
    If no face is detected, retries up to `max_retries` times.
    """
    last_result = (None, None, None, None)

    for attempt in range(1, max_retries + 1):
        print(f"\n[RETRY] Face Recognition attempt {attempt}/{max_retries}")

        try:
            # --- PREVIEW PHASE ---
            print(f"[CAMERA] Starting preview (attempt {attempt}) for {preview_duration}s...")
            init_camera_preview()
            time.sleep(preview_duration)

            # --- CAPTURE PHASE ---
            print(f"[CAMERA] Switching to still mode for capture (attempt {attempt})...")
            switch_to_still_mode()
            time.sleep(0.5)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_img_path = os.path.join(UPLOAD_FOLDER, f"face_attempt_{attempt}_{timestamp}.jpg")

            frame = picam2.capture_array()
            cv2.imwrite(face_img_path, frame)
            print(f"[CAPTURE] Saved face image: {face_img_path}")

            # --- RUN FACE RECOGNITION ---
            annotated_img, face_name, model_time, compare_time = perform_face_recognition(frame)
            last_result = (annotated_img, face_name, model_time, compare_time)

            # --- SUCCESS CASE ---
            if face_name and str(face_name).strip():
                print(f"[SUCCESS] Face recognized: {face_name} (attempt {attempt})")
                play_audio("Okay__Face_Captured.mp3")
                return last_result

            # --- FAILURE CASE ---
            print(f"[FAIL] No recognizable face detected (attempt {attempt})")
            # play_audio("Okay__No_face_detected__try_again.mp3")
            play_audio("Okay__No_Face_Recognized.mp3")

            if attempt < max_retries:
                print(f"[RETRY] Waiting {delay_between_attempts}s before next attempt...")
                time.sleep(delay_between_attempts)

        except Exception as e:
            print(f"[ERROR] Face capture/recognition failed on attempt {attempt}: {e}")
            if attempt < max_retries:
                print(f"[RETRY] Retrying in {delay_between_attempts}s...")
                time.sleep(delay_between_attempts)

    print("[ERROR] Face could not be recognized after all retry attempts.")
    play_audio("face_recognition_failed.mp3")
    return last_result


# =====================================================
# ================ MAIN CAPTURE LOOP ==================
# =====================================================

def capture_loop():
    play_audio("Okay__System_is_ready.mp3")
    ser.flush()
    while True:
        if ser.in_waiting > 0:
            msg = ser.readline().decode().strip()
            if not msg:
                continue
            print(f"[RPi] Received: {msg}")

            if msg == "vehicleDetected":
                print("[DEBUG] Vehicle detected, starting process...")

                run_timings = {
                    "step_load_face_embeddings": last_load_faces_time
                }

                play_audio("Okay__Vehicle_Detected.mp3")

                # --- Fetch plate image ---
                plate_img = fetch_plate_from_pi_zero()
                if plate_img is None:
                    print("[ERROR] No image received from Pi Zero.")
                    play_audio("Okay__No_Plate_Detected__try_again.mp3")
                    ser.write(b"resetCycle\n")
                    ser.flush()
                    continue

                plate_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                plate_path = os.path.join(UPLOAD_FOLDER, f"plate_{plate_ts}.jpg")
                cv2.imwrite(plate_path, plate_img)

                # --- LPR with retry ---
                plate_img, plate_text, yolov8_time, easyocr_time = perform_lpr_with_retry(plate_img)
                run_timings["step_yolov8_onnx"] = round(yolov8_time, 4) if yolov8_time else None
                run_timings["step_easyocr"] = round(easyocr_time, 4) if easyocr_time else None
                
                if not plate_text:
                    print("[ERROR] No plate detected after retries.")
                    # play_audio("error_no_plate_detected.mp3")
                    play_audio("Okay__No_Plate_Detected__try_again.mp3")
                    # ser.write(b"isNotAuthorized\n")
                    ser.write(b"resetCycle\n")
                    ser.flush()
                    continue

                print(f"[INFO] Plate OCR Result: {plate_text}")

                try:
                    print("[DEBUG] Checking if detected plate is registered in Supabase...")
                    response = supabase.table("plate_numbers_with_users") \
                        .select("plate_number") \
                        .eq("plate_number", plate_text) \
                        .execute()

                    # If no record found, reset the cycle immediately
                    if not response.data or len(response.data) == 0:
                        print(f"[WARNING] Plate {plate_text} is NOT registered. Resetting cycle.")
                        play_audio("Okay__Plate_not_registered__try_again.mp3")

                        # Reset Arduino cycle
                        ser.write(b"resetCycle\n")
                        ser.flush()
                        continue

                    print(f"[INFO] Plate {plate_text} is registered. Proceeding to face recognition.")

                except Exception as e:
                    print(f"[ERROR] Plate validation check failed: {e}")
                    play_audio("Okay__Error_validating_plate__try_again.mp3")
                    ser.write(b"resetCycle\n")
                    ser.flush()
                    continue

                # --- Distance Validation ---
                stable_count, bad_count, max_bad, attempts = 0, 0, 10, 0
                print("[DEBUG] Starting distance validation...")
                ser.write(b"startSendDistance\n")
                play_audio("Okay__Face_the_camera_around_1_meter_away.mp3")

                while stable_count < 4 and bad_count < max_bad and attempts < 400:
                    if ser.in_waiting > 0:
                        dist_msg = ser.readline().decode().strip()
                        if dist_msg.isdigit():
                            distance = int(dist_msg)
                            print(f"[DEBUG] Distance: {distance}")
                            if 50 <= distance <= 170:
                                stable_count += 1
                                print(f"[DEBUG] Stable count: {stable_count}")
                            else:
                                bad_count += 1
                                print(f"[DEBUG] Bad count: {bad_count}")
                    attempts += 1
                    time.sleep(0.1)

                ser.write(b"endSendDistance\n")

                if stable_count < 4:
                    print("[WARNING] Distance validation failed.")
                    ser.write(b"resetCycle\n")
                    ser.flush()
                    continue
                
                print("[DEBUG] Distance validation passed, capturing face...")

                play_audio("Okay_Your_photo_will_be_taken_in_three____two____one___.mp3")

                # --- Face Preview ---
                init_camera_preview()

                # --- Capture face still image ---
                switch_to_still_mode()
                time.sleep(0.2)
                face_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_path = os.path.join(UPLOAD_FOLDER, f"face_{face_ts}.jpg")

                start_capture = time.time()
                picam2.capture_file(face_path)
                capture_elapsed = time.time() - start_capture
                run_timings["step_capture_rpi_cam3"] = round(capture_elapsed, 4)
                print(f"[PERF] Step capture RPi Cam 3 took {capture_elapsed:.4f} sec")

                # --- Face Recognition with retry ---
                face_img = cv2.imread(face_path)
                face_img, face_name, arcface_model_time, arcface_compare_time = perform_face_recognition_with_retry()
                run_timings["step_arcface_model"] = round(arcface_model_time, 4) if arcface_model_time else None
                run_timings["step_arcface_comparison"] = round(arcface_compare_time, 4) if arcface_compare_time else None
            
                if not face_name:
                    print("[ERROR] No face detected after retries.")
                    # play_audio("Okay__No_face_detected__try_again.mp3")
                    play_audio("Okay__No_Face_Recognized.mp3")
                    # ser.write(b"isNotAuthorized\n")
                    ser.write(b"resetCycle\n")
                    ser.flush()
                    continue

                play_audio("license_plate_and_driver_s_face_recognition_complete.mp3")

                # # --- Display captured frames ---
                # cv2.imshow("Plate", resize(plate_img))
                # cv2.imshow("Face", resize(face_img))
                # while cv2.waitKey(1) & 0xFF != ord('q'):
                #     pass
                # cv2.destroyAllWindows()

                # --- Validate results ---
                is_authorized, validation_msg = validate(plate_text, face_name)
                print("[DEBUG] Sending results to Supabase log...")

            
                send_to_supabase(plate_text, face_name, is_authorized, timings=run_timings)

                # --- Display captured frames (added delay) ---
                time.sleep(7)

                # --- Gate control ---
                if is_authorized:
                    print("[DEBUG] Authorized, opening gate...")
                    ser.write(b"isAuthorized\n")
                    ser.flush()
                    play_audio("Okay__You_are_authorized__Welcome_to_Samar_State_University.mp3")
                else:
                    print("[DEBUG] Not authorized, rejecting...")
                    ser.write(b"isNotAuthorized\n")
                    ser.flush()
                    play_audio("Okay__You_are_not_authorized__Register_first_using_valpir_mobile_app.mp3")

                # --- Reset Cycle ---
                time.sleep(5)
                ser.write(b"resetCycle\n")
                ser.flush()
                print("[DEBUG] Cycle reset complete")


if __name__ == '__main__':
    capture_loop()
