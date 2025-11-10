# 🎭 Deepfake Detection Project

This project is an **AI-powered Deepfake Video Detection System** built in **Google Colab** using **PyTorch** and **OpenCV**.  
It detects whether a video is *real* or *fake* by analyzing faces extracted from video frames.

---

## 🚀 Features
- Extracts face frames using **OpenCV Haar Cascades**
- Trains a **ResNet-18** CNN for binary classification (real vs fake)
- Supports **video-level prediction** by averaging frame probabilities
- Includes **Grad-CAM visualizations** for explainability
- Built and tested entirely in **Google Colab (GPU)**

---

## 🧠 Model Details
- **Architecture:** ResNet-18 (pretrained on ImageNet)  
- **Optimizer:** Adam  
- **Loss Function:** CrossEntropyLoss  
- **Input Size:** 224×224 face crops  
- **Dataset:** 53 real videos + 53 fake videos (Mendeley dataset)

---

## 🧩 Project Structure
