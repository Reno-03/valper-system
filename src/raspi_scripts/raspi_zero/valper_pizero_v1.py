from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time

# ZeroConf
import time, socket, requests
from zeroconf import Zeroconf, ServiceInfo
import signal
import sys

SERVICE_TYPE = "_demo._tcp.local."
SERVICE_NAME = "pizero._demo._tcp.local."
FLASK_PORT = 5003

app = Flask(__name__)
picam2 = Picamera2()

# ======================================================
# Camera setup (same as your original)
# ======================================================
full_res_cfg = picam2.create_still_configuration(
    main={"format": "RGB888", "size": (3280, 2464)}  # Pi Camera V2 full res
)
picam2.configure(full_res_cfg)
picam2.set_controls({"AfMode": 2})  # Autofocus
time.sleep(0.5)
picam2.set_controls({"AfTrigger": 0})
picam2.start()

@app.route('/capture')
def capture():
    # Capture full 4:3 frame
    frame = picam2.capture_array()

    # Convert RGB → BGR for OpenCV (optional, depending on how you use it)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ---- Crop to 16:9 without distortion ----
    h, w, _ = frame.shape
    target_aspect = 16 / 9
    new_h = int(w / target_aspect)

    if new_h <= h:
        # crop vertically
        crop_y = (h - new_h) // 2
        frame = frame[crop_y:crop_y + new_h, :]
    else:
        # fallback crop horizontally (rare case)
        new_w = int(h * target_aspect)
        crop_x = (w - new_w) // 2
        frame = frame[:, crop_x:crop_x + new_w]

    # Resize to 1920x1080
    frame = cv2.resize(frame, (1920, 1080))

    # Encode as JPEG
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return Response("Camera error", status=500)
    return Response(
        jpeg.tobytes(),
        mimetype='image/jpeg',
        headers={"Cache-Control": "no-store"}
    )

@app.route('/')
def root():
    """Basic health check endpoint"""
    return {"status": "ok", "device": "pizero"}


# ======================================================
# Zeroconf (FIXED)
# ======================================================
def get_ip():
    """Get local LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def register_service():
    """Register this Pi Zero as a Zeroconf service."""
    ip_address = get_ip()
    print(f"[PiZero] Local IP: {ip_address}")

    info = ServiceInfo(
        SERVICE_TYPE,
        SERVICE_NAME,
        addresses=[socket.inet_aton(ip_address)],
        port=FLASK_PORT,
        properties={b"device": b"pizero"},
        server="pizero.local."
    )

    zeroconf = Zeroconf()
    try:
        zeroconf.register_service(info)
        print(f"[PiZero] Zeroconf service registered as {SERVICE_NAME} ({ip_address}:{FLASK_PORT})")
    except Exception as e:
        print(f"[PiZero][ERROR] Zeroconf registration failed: {e}")

    # Graceful shutdown
    def shutdown(*args):
        print("\n[PiZero] Unregistering Zeroconf service...")
        try:
            zeroconf.unregister_service(info)
            zeroconf.close()
        except Exception:
            pass
        sys.exit(0)

    import signal
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    return zeroconf


if __name__ == '__main__':
    # Register Zeroconf
    zeroconf = register_service()

    print(f"[INFO] Pi Zero Camera Server running on http://0.0.0.0:{FLASK_PORT}/capture")
    print("[INFO] Press Ctrl+C to stop.")
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False)

    # On exit, cleanup Zeroconf
    try:
        zeroconf.unregister_service(info)
        zeroconf.close()
    except Exception:
        pass
