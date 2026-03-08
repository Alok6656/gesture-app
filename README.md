# 🖐 GestureAI — Hand Language Interpreter

A real-time hand gesture recognition web app powered by Python (Flask) + MediaPipe + OpenCV.

---

## 📁 Project Structure

```
gesture-app/
│
├── app.py                  ← Flask backend (main server)
│   ├── MediaPipe setup      Hand landmark detection
│   ├── Gesture logic        12+ gesture classifiers
│   └── Routes:
│       ├── GET  /           Serve the web UI
│       ├── POST /analyze    Accept base64 image → return gesture
│       └── GET  /gestures-guide  Return gesture dictionary
│
├── templates/
│   └── index.html          ← Single-page frontend (HTML + CSS + JS)
│       ├── Camera panel     Live webcam feed with scan overlay
│       ├── Result panel     Detected gesture + confidence bar
│       ├── History log      Last 20 recognized gestures
│       └── Gesture guide    Reference card for all 12 gestures
│
├── static/                 ← (optional) Static assets
│
└── requirements.txt        ← Python dependencies
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python app.py
```

### 3. Open the browser
```
http://localhost:5000
```

---

## 🖐 Recognized Gestures

| Gesture        | Meaning              |
|----------------|----------------------|
| ✊ Fist         | Stop / Closed fist   |
| 🖐 Open Hand   | High five / Open     |
| 👍 Thumbs Up   | Good / Approve       |
| 👎 Thumbs Down | Bad / Disapprove     |
| ☝️ Point Up    | One / Attention      |
| ✌️ Peace Sign  | Victory / Two        |
| 👌 OK Sign     | Perfect / Okay       |
| 🤘 Rock On     | Metal / Rock         |
| 🤙 Call Me     | Hang Loose / Call    |
| 🤟 I Love You  | ILY                  |
| 🤞 Crossed     | Good Luck            |
| 🖖 Vulcan      | Live Long and Prosper|

---

## ⚙️ How It Works

```
Browser Webcam
     │
     ▼ (base64 JPEG via fetch POST)
Flask /analyze endpoint
     │
     ▼
MediaPipe Hands
  → 21 3D landmarks per hand
     │
     ▼
Gesture Classifier
  → Checks finger extension states
  → Calculates landmark distances
  → Returns gesture label
     │
     ▼
Response JSON
  {
    "gesture": "✌️ Peace / Victory",
    "hands_detected": 1,
    "details": [{"hand": "Right", "gesture": "..."}],
    "annotated_image": "data:image/jpeg;base64,..."
  }
     │
     ▼
Browser displays result + annotated image with skeleton overlay
```

---

## 🛠 Modes

- **SNAP mode**: Click ⊙ CAPTURE manually to analyze one frame
- **LIVE mode**: Auto-analyzes every 1.5 seconds (continuous recognition)

---

## 📦 Tech Stack

| Layer     | Technology          |
|-----------|---------------------|
| Backend   | Python 3.x + Flask  |
| AI/ML     | MediaPipe Hands     |
| Vision    | OpenCV              |
| Frontend  | HTML5 + CSS + JS    |
| Camera    | WebRTC getUserMedia |