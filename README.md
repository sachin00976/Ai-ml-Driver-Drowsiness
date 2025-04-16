# Driver Drowsiness Detection System

This project is a **real-time driver drowsiness detection system** using computer vision techniques. It monitors eye activity using a webcam and alerts the driver with an alarm and a simple arithmetic challenge when drowsiness is detected.

## 🔍 Features

- Real-time face and eye detection using **dlib** and **OpenCV**
- Computes **Eye Aspect Ratio (EAR)** to detect closed eyes
- Triggers an alert system with a beeping sound if drowsiness is detected
- Displays a simple math challenge (e.g., `12 + 17 = ?`) to confirm driver awareness
- Uses **CLAHE** to enhance eye region detection
- Includes face tracking for better performance

## 📹 Demo

*Add a GIF or video link here showing the project in action.*

## 🛠️ Requirements

- Python 3.x
- OpenCV
- dlib
- pygame
- scipy
- numpy

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pranjal0241/drowsiness-detection.git
   cd drowsiness-detection
