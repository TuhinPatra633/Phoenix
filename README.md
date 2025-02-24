# AI Assistant for People with Disabilities

## Team Phoenix - Exathon 2025

### Team Members
- Pankaj Sadhukhan
- Saikat Kumar Ghosh
- Tuhin Patra

## Overview
An AI-driven accessibility tool designed to assist individuals with disabilities. The system integrates multiple AI models to provide real-time navigation, communication enhancement, hands-free interaction, and mental health assessment.

## Problem Statement
1. **Empowering the Visually Impaired**: Lack of real-time navigation assistance affects mobility and independence.
2. **Enhancing Communication for the Hearing & Speech Impaired**: Over-reliance on spoken or written language creates barriers in education, work, and social engagement.
3. **Enabling Hands-Free Interaction for Mobility-Impaired Users**: Traditional input methods (keyboard/touchscreen) are inaccessible to certain individuals.
4. **Addressing Mental Health Issues**: Rising mental health concerns demand efficient and early-stage detection models.

## Key Features
- **Visual Model**: Object detection, depth estimation, and audio feedback for navigation assistance.
- **Speech-to-Text & Text-to-Speech**: Enables seamless communication for hearing and speech-impaired individuals.
- **Gesture Recognition**: Allows hands-free interaction using predefined gestures.
- **Mental Health Prediction Model**: Detects early signs of mental health issues.
- **Web Application**: User-friendly interface integrating all functionalities.

## Implementation Details
### 1. **Visual Model**
- Captures real-time frames via a camera.
- Detects objects using **YOLOv8 segmentation**.
- Estimates depth with **MiDaS**.
- Provides structured audio feedback using **Google Text-to-Speech (gTTS)**.

### 2. **Speech-to-Text & Text-to-Speech Model**
- Uses **Whisper AI** for speech-to-text conversion.
- Converts text to speech using **Tacotron 2** or **Google TTS**.

### 3. **Gesture Recognition Model**
- Utilizes **Mediapipe Hand Tracking** for gesture-based commands:
  - **Index Finger** → Cursor Movement
  - **Index + Middle Finger** → Right Click
  - **Pinch** → Left Click
  - **Open Palm (Up/Down)** → Scroll

### 4. **Mental Health Prediction Model**
- Uses a dataset from the **Wellcome Global Monitor 2020**.
- Models trained with **Random Forest, SVM, and MLP**.
- Best accuracy (**78%**) achieved using **Random Forest + SMOTE**.

### 5. **Web Application**
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Features**:
  - AI-powered object detection and navigation.
  - Speech and gesture-based interaction.
  - Mental health assessment module.


## Technologies Used
- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **Models**: YOLOv8, MiDaS, Whisper AI, Tacotron 2, Mediapipe
- **Web Technologies**: Flask, HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy, OpenCV

## Future Improvements
- Enhance real-time processing efficiency.
- Expand language support for STT and TTS models.
- Improve gesture recognition accuracy.
- Integrate emotion detection for mental health assessment.

