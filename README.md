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
deepfake_project/
│
├── DeepFakeDetection.ipynb # Main Colab Notebook
├── best_model_resnet18.pth # Saved Model File
├── README.md # Project Documentation
└── requirements.txt # Required Dependencies

---

## ⚙️ How to Run in Google Colab
1. **Open the notebook** `DeepFakeDetection.ipynb` in Google Colab.  
2. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Prepare your dataset inside Google Drive like this:

deepfake_dataset/
├── real/
└── fake/


Run all cells in order:

Face extraction using Haar Cascade

Train ResNet-18 model

Evaluate test accuracy & generate Grad-CAMs

Upload a new video to test predictions

At the end, upload any new video and the model will predict whether it’s REAL or FAKE.

📊 Example Output
{'predicted_label': 'fake', 'avg_prob_fake': 0.84, 'frames_used': 8}


Grad-CAM Visualization Example:
(Shows what part of the face the model focused on)

🧩 Requirements

Install all dependencies using:

pip install -r requirements.txt
