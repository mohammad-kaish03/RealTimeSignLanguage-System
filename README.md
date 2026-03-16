# Real-Time Sign Language Recognition System

A machine learning and computer vision based system that recognizes sign language gestures in real time using a webcam and converts them into readable text. The project aims to reduce communication barriers between sign language users and non-signers.

---

## 📌 Features

- Real-time hand gesture detection using webcam
- Hand landmark extraction using MediaPipe
- Machine learning based gesture classification
- Automatic conversion of gestures into readable text
- Sentence generation from multiple gestures
- Interactive web interface using Streamlit

---

## 🧠 Technologies Used

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Pandas
- Streamlit
- Joblib

---

## 🏗️ System Architecture

The system follows a machine learning pipeline:

1. Video input captured through webcam  
2. Hand detection using MediaPipe  
3. Hand landmark extraction  
4. Feature vector generation  
5. Gesture classification using a trained model  
6. Text generation from predicted gestures  

---

## ⚙️ Implementation Steps

- Collected gesture data using webcam
- Extracted hand landmarks using MediaPipe
- Created a dataset and stored features in CSV format
- Trained a Random Forest classification model
- Implemented real-time prediction system
- Built a Streamlit interface for user interaction

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/sign-language-recognition.git
```

### 2️⃣ Navigate to the project directory

```bash
cd sign-language-recognition
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
streamlit run app.py
```

---

## 📊 Results

- Successful real-time detection of sign language gestures
- Accurate classification of predefined gesture patterns
- Smooth performance with standard webcam
- Instant conversion of gestures into readable text

---

## 🔮 Future Improvements

- Support for dynamic sign language gestures
- Deep learning based gesture recognition
- Text-to-speech integration
- Mobile application version
- Multi-language sign recognition

---

## 👨‍💻 Authors

**Mohammad Kaish Ansari**

GitHub: https://github.com/mohammad-kaish03  
LinkedIn: https://www.linkedin.com/in/mohammad-kaish-ansari

---

## 📚 References

- Python Official Documentation  
- MediaPipe Documentation  
- Scikit-learn Documentation  
- Streamlit Framework  
