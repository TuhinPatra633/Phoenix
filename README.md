# 🚀 AI Assistant for People with Disabilities

## 🌟 Team Phoenix - Exathon 2025

### 🏆 Team Members
- **Pankaj Sadhukhan**
- **Saikat Kumar Ghosh**
- **Tuhin Patra**

## 🌍 Overview
An **AI-driven accessibility tool** designed to empower individuals with disabilities! Our system integrates cutting-edge AI models to provide **real-time navigation, communication enhancement, hands-free interaction, and mental health assessment.**

## 🔥 Why This Matters
💡 **Empowering the Visually Impaired**: Navigate the world with confidence using real-time object detection and audio feedback.

🗣️ **Enhancing Communication for the Hearing & Speech Impaired**: Convert speech to text and vice versa seamlessly.

🤖 **Hands-Free Interaction for Mobility-Impaired Users**: Control devices effortlessly using intuitive hand gestures.

💙 **Mental Health Awareness**: Early detection through AI-driven analysis, ensuring timely support.

## 🚀 Key Features
✅ **Visual Model** – Object detection, depth estimation, and audio feedback for smooth navigation.

✅ **Speech-to-Text & Text-to-Speech** – Seamless communication with Whisper AI and Tacotron 2.

✅ **Gesture Recognition** – Control devices hands-free with simple gestures.

✅ **Mental Health Prediction Model** – AI-powered analysis to detect early mental health concerns.

✅ **Web Application** – A user-friendly interface integrating all functionalities.

## 🛠️ Implementation Details
### 🎯 1. **Visual Model**
- 📸 Captures real-time frames via a camera.
- 🏷️ Detects objects using **YOLOv8 segmentation**.
- 📏 Estimates depth with **MiDaS**.
- 🔊 Provides structured audio feedback using **Google Text-to-Speech (gTTS)**.

### 🗣️ 2. **Speech-to-Text & Text-to-Speech**
- 🎙️ **Whisper AI** for speech-to-text conversion.
- 🔉 **Tacotron 2** or **Google TTS** for natural speech synthesis.

### ✋ 3. **Gesture Recognition Model**
- 🖐️ Uses **Mediapipe Hand Tracking** for intuitive gestures:
  - ☝️ **Index Finger** → Cursor Movement
  - ✌️ **Index + Middle Finger** → Right Click
  - 🤏 **Pinch** → Left Click
  - 🖐️ **Open Palm (Up/Down)** → Scroll

### 🧠 4. **Mental Health Prediction Model**
- 📊 Trained on **Wellcome Global Monitor 2020** dataset.
- 🏆 Best accuracy (**78%**) achieved using **Random Forest + SMOTE**.

### 🌐 5. **Web Application**
- **Backend**: Flask 🐍
- **Frontend**: HTML, CSS, JavaScript 🎨
- **Features**:
  - 🎯 AI-powered object detection and navigation.
  - 💬 Speech & gesture-based interaction.
  - ❤️ Mental health assessment module.


## 🏗️ Tech Stack
🔹 **Deep Learning**: PyTorch, TensorFlow
🔹 **Models**: YOLOv8, MiDaS, Whisper AI, Tacotron 2, Mediapipe
🔹 **Web**: Flask, HTML, CSS, JavaScript
🔹 **Data Processing**: Pandas, NumPy, OpenCV

## 🚀 Future Roadmap
🚀 Improve real-time processing efficiency.
🌎 Expand language support for STT and TTS models.
🖐️ Enhance gesture recognition accuracy.
❤️ Integrate emotion detection for mental health assessment.


🌟 _Let's make the world more accessible for everyone!_
