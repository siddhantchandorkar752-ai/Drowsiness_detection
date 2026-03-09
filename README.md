@'
# 🚗 Driver Drowsiness Detection System
### Real-Time CNN-Powered Eye State Monitoring

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![CNN](https://img.shields.io/badge/Model-CNN-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A real-time drowsiness detection system that monitors driver eye states through a webcam feed and triggers an immediate audio alert when prolonged eye closure is detected — preventing accidents before they happen.

---

## 🎯 The Problem
```
Every year, 1.35 million people die in road accidents.
20% of all serious accidents are fatigue-related.
A driver closing eyes for just 2 seconds at 100km/h
travels 55 meters completely blind.
```

**This system detects that 2-second window — and acts.**

---

## ⚡ How It Works
```
Webcam Feed
     │
     ▼
Haar Cascade ──► Face Detection
     │
     ▼
Haar Cascade ──► Eye Region Extraction
     │
     ▼
CNN Model ──────► Open / Closed Classification
     │
     ▼
Alert Engine ───► 🔊 Alarm if eyes closed > threshold
```

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| 👁️ Real-Time Monitoring | Live webcam feed analysis at 30fps |
| 🧠 CNN Classification | Custom trained model — Open vs Closed eyes |
| 🔊 Instant Alert | Audio alarm triggers within milliseconds |
| 📡 Haar Cascades | Robust face + eye localization |
| ⚡ Lightweight | Runs on CPU — no GPU required |

---

## 📁 Project Structure
```
Drowsiness_detection/
├── haar cascade files/
│   ├── haarcascade_frontalface_alt.xml
│   └── haarcascade_eye_tree_eyeglasses.xml
├── models/
│   └── cnnCat2.h5              # Trained CNN model
├── drowsinessdetection.py      # Main inference script
├── model.py                    # CNN architecture + training
├── alarm.wav                   # Alert audio
└── README.md
```

---

## 🚀 Quick Start

### Install Dependencies
```bash
git clone https://github.com/siddhantchandorkar752-ai/drowsiness-detection.git
cd drowsiness-detection
pip install opencv-python tensorflow keras pygame numpy
```

### Run
```bash
python drowsinessdetection.py
```

---

## 🧠 Model Architecture
```
Input: 24x24 Grayscale Eye Image
     │
Conv2D(32) → ReLU → MaxPool
     │
Conv2D(64) → ReLU → MaxPool
     │
Conv2D(128) → ReLU → MaxPool
     │
Flatten → Dense(128) → Dropout(0.5)
     │
Dense(2) → Softmax
     │
Output: [Open, Closed]
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Computer Vision**: OpenCV + Haar Cascades
- **Deep Learning**: TensorFlow / Keras CNN
- **Audio**: Pygame
- **Model**: Custom trained `cnnCat2.h5`

---

## 📊 Real World Impact
```
Without System          With System
─────────────────       ──────────────────
Driver falls asleep  →  Alert at first sign
Accident occurs      →  Driver wakes up
Injury / Death       →  Life saved ✅
```

---

## 👨‍💻 Author

**Siddhant Chandorkar**
- GitHub: [@siddhantchandorkar752-ai](https://github.com/siddhantchandorkar752-ai)

---

## 📄 License
MIT License — Free to use, modify, and distribute.
'@ | Set-Content -Path README_drowsiness.md -Encoding UTF8
