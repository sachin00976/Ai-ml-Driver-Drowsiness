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

![Screenshot (2)](https://github.com/user-attachments/assets/f0aeab1e-1d13-4af2-969d-3ca67b4d9dd7)![Screenshot (1)](https://github.com/user-attachments/assets/b086acc9-d881-4568-bcc2-7395319cbfb5)


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
