# 🖐 GestureAI — Hand Language Interpreter

A real-time hand gesture recognition web application that uses your webcam 
to detect hand gestures and convert them into text — instantly, privately, 
and without any API key.

## ✨ Features
- 🎥 Live webcam gesture detection
- 🧠 AI-powered hand landmark tracking (21 points per hand) 
- ✋ Recognizes 12+ gestures (Thumbs Up, Peace, OK, Rock On & more)
- 🔒 100% local — no cloud, no API key, no data sent anywhere
- ⚡ Real-time response in under 1 second
- 🌐 Works in any browser — no app install needed
- 📸 Snap mode & Live mode

## 🛠 Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python 3 + Flask |
| AI / ML | MediaPipe Hands |
| Vision | OpenCV |
| Frontend | HTML5 + CSS3 + JavaScript |
| Camera | WebRTC getUserMedia |

## 🚀 Quick Start
```bash
pip install flask mediapipe opencv-python numpy
python app.py
# Open http://localhost:5000
```

## 🖐 Supported Gestures
✊ Fist · 🖐 Open Hand · 👍 Thumbs Up · 👎 Thumbs Down · ☝️ Point
✌️ Peace · 👌 OK · 🤘 Rock On · 🤙 Call Me · 🤟 ILY · 🤞 Crossed · 🖖 Vulcan

## 📁 Project Structure
```
gesture-app/
├── app.py              ← Flask backend + gesture classifier
├── templates/
│   └── index.html      ← Frontend UI
├── hand_landmarker.task← MediaPipe model (auto-downloaded)
└── requirements.txt    ← Dependencies
```
```

---

**GitHub Topics** (add these as tags on your repo page):
```
python  flask  mediapipe  opencv  gesture-recognition  
hand-tracking  computer-vision  sign-language  
machine-learning  real-time  webcam  hackathon
