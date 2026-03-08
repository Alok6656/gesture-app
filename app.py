"""
GestureAI Backend — mediapipe 0.10.x (Tasks API)
No API key required. Model file (~9 MB) is downloaded once from Google's
public storage on first run, then cached in the project folder.

Run:  python app.py
Open: http://localhost:5000
"""

import os, sys, base64, math, urllib.request
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_FILE = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    """Download the hand landmarker model once; cache it locally."""
    if os.path.exists(MODEL_FILE):
        print(f"[INFO] Model found: {MODEL_FILE}")
        return True
    print("[INFO] hand_landmarker.task not found — downloading (~9 MB) ...")
    print(f"       From: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[INFO] Download complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("[HINT]  Download manually from:")
        print(f"        {MODEL_URL}")
        print(f"        and place as: {MODEL_FILE}")
        return False

detector = None

def init_mediapipe():
    global detector
    if not ensure_model():
        return False
    try:
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python import vision

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_FILE),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = HandLandmarker.create_from_options(options)
        print("[INFO] MediaPipe HandLandmarker ready.")
        return True
    except Exception as e:
        print(f"[ERROR] MediaPipe init failed: {e}")
        return False

init_mediapipe()

WRIST=0; THUMB_TIP=4; THUMB_IP=3; THUMB_MCP=2
INDEX_TIP=8; INDEX_PIP=6; INDEX_MCP=5
MIDDLE_TIP=12; MIDDLE_PIP=10
RING_TIP=16; RING_PIP=14
PINKY_TIP=20; PINKY_PIP=18

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def thumb_up_check(lm, hand_label):

    if hand_label == "Right":
        return lm[THUMB_TIP].x < lm[THUMB_IP].x
    return lm[THUMB_TIP].x > lm[THUMB_IP].x

def dist2d(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def classify(lm, hand_label):
    th = thumb_up_check(lm, hand_label)
    ix = finger_up(lm, INDEX_TIP,  INDEX_PIP)
    mi = finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
    ri = finger_up(lm, RING_TIP,   RING_PIP)
    pi = finger_up(lm, PINKY_TIP,  PINKY_PIP)
    n  = sum([th, ix, mi, ri, pi])

    if n == 0: return "✊ Fist / Stop"
    if n == 5: return "🖐 Open Hand / High Five"

    if th and not ix and not mi and not ri and not pi:
        return "👍 Thumbs Up" if lm[THUMB_TIP].y < lm[WRIST].y else "👎 Thumbs Down"

    if not th and ix and not mi and not ri and not pi:
        return "☝️ Pointing Up"

    if not th and ix and mi and not ri and not pi:
        gap = abs(lm[INDEX_TIP].x - lm[MIDDLE_TIP].x)
        return "✌️ Peace / Victory" if gap > 0.03 else "🤞 Crossed Fingers"

    if dist2d(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.06 and mi and ri and pi:
        return "👌 OK / Perfect"

    if not th and ix and not mi and not ri and pi:
        return "🤘 Rock On / Metal"

    if th and not ix and not mi and not ri and pi:
        return "🤙 Call Me / Hang Loose"

    if th and ix and not mi and not ri and pi:
        return "🤟 I Love You"

    if not th and ix and mi and ri and not pi:
        return "3️⃣ Three Fingers"

    if not th and ix and mi and ri and pi:
        gap = abs(lm[MIDDLE_TIP].x - lm[RING_TIP].x)
        return "🖖 Vulcan Salute" if gap > 0.05 else "4️⃣ Four Fingers"

    if not th and not ix and not mi and not ri and pi:
        return "🤙 Pinky Up"

    return f"🤔 Unknown ({n} fingers up)"

BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_skeleton(img, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in BONES:
        cv2.line(img, pts[a], pts[b], (0, 255, 180), 2)
    for pt in pts:
        cv2.circle(img, pt, 5, (0, 220, 255), -1)
        cv2.circle(img, pt, 7, (255, 255, 255), 1)

def analyse(bgr_frame):
    if detector is None:
        return "⚠️ Model not loaded", [], 0, bgr_frame

    import mediapipe as mp
    h, w   = bgr_frame.shape[:2]
    rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    canvas = bgr_frame.copy()

    if not result.hand_landmarks:
        return "No hands detected", [], 0, canvas

    details  = []
    gestures = []

    for i, hand_lm in enumerate(result.hand_landmarks):
        draw_skeleton(canvas, hand_lm, h, w)
        label = "Right"
        if result.handedness and i < len(result.handedness):
            label = result.handedness[i][0].category_name
        g = classify(hand_lm, label)
        details.append({"hand": label, "gesture": g})
        gestures.append(g)

    if len(gestures) == 1:
        combined = gestures[0]
    else:
        combined = "  +  ".join(gestures)
        if all("Thumbs Up"  in g for g in gestures): combined = "👍👍 Double Thumbs Up!"
        elif all("Peace"    in g for g in gestures): combined = "✌️✌️ Double Peace!"
        elif all("Open Hand" in g for g in gestures): combined = "🙌 Celebration!"

    return combined, details, len(result.hand_landmarks), canvas

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image", "success": False}), 400

        raw = data["image"]
        if "," in raw:
            raw = raw.split(",", 1)[1]
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(raw), np.uint8),
            cv2.IMREAD_COLOR
        )
        if frame is None:
            return jsonify({"error": "Bad image data", "success": False}), 400

        gesture, details, n_hands, annotated = analyse(frame)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        return jsonify({
            "success":         True,
            "gesture":         gesture,
            "details":         details,
            "hands_detected":  n_hands,
            "annotated_image": ann_b64,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/gestures-guide")
def gestures_guide():
    return jsonify([
        {"gesture": "✊ Fist",        "meaning": "Stop / Closed fist"},
        {"gesture": "🖐 Open Hand",   "meaning": "High five / Open"},
        {"gesture": "👍 Thumbs Up",   "meaning": "Good / Approve"},
        {"gesture": "👎 Thumbs Down", "meaning": "Bad / Disapprove"},
        {"gesture": "☝️ Point Up",    "meaning": "One / Attention"},
        {"gesture": "✌️ Peace",       "meaning": "Victory / Two"},
        {"gesture": "👌 OK Sign",     "meaning": "Perfect / Okay"},
        {"gesture": "🤘 Rock On",     "meaning": "Rock / Metal"},
        {"gesture": "🤙 Call Me",     "meaning": "Hang Loose"},
        {"gesture": "🤟 ILY",         "meaning": "I Love You"},
        {"gesture": "🤞 Crossed",     "meaning": "Good Luck"},
        {"gesture": "🖖 Vulcan",      "meaning": "Live Long and Prosper"},
    ])

@app.route("/status")
def status():
    return jsonify({
        "ready":       detector is not None,
        "model_file":  os.path.exists(MODEL_FILE),
        "model_path":  MODEL_FILE,
    })

if __name__ == "__main__":
    print("\n🖐  GestureAI — mediapipe", end=" ")
    import mediapipe; print(mediapipe.__version__)
    print(f"   Model  : {'✅ Ready' if detector else '❌ Not loaded'}")
    print("   URL    : http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)