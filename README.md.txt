# 🤟 Real-Time Sign Language Recognition System

## 📌 Overview
A real-time computer vision application that detects hand gestures and converts them into meaningful words and sentences using Machine Learning.

---

## 🎯 Problem Statement
Communication barriers exist for deaf and mute individuals in real-time interactions. This system helps bridge that gap by enabling gesture-based communication.

---

## 🧠 Why This Project Matters

This project goes beyond a basic ML model and implements a complete pipeline:

- Custom dataset collection system  
- Real-time gesture recognition  
- Sentence generation logic  
- Interactive user interface  

👉 This makes it a full end-to-end ML application.

---

## ⚙️ How It Works

1. User shows hand gesture to webcam  
2. MediaPipe extracts 63 hand landmark features  
3. Machine Learning model predicts gesture  
4. Prediction buffer smooths output  
5. Words are combined into a sentence  

---

## 📊 Results

- ✅ Accuracy: **90–92%**
- ⚡ Real-time performance: **15–20 FPS**
- 🚀 Reduced latency by **25%**

---

## 💡 Key Features

- Real-time gesture detection  
- Sentence formation system  
- Prediction smoothing (deque buffer)  
- Interactive Streamlit UI  

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- Scikit-learn  
- Streamlit  

---

## 📸 Demo

![Demo 1](images/demo1.png)
![Demo 2](images/demo2.png)
![Demo 3](images/demo3.png)

---

## 🚀 Installation & Usage

```bash
git clone https://github.com/mohammad-kaish03/RealTimeSignLanguage-System.git
cd RealTimeSignLanguage-System

pip install -r requirements.txt
streamlit run app.py