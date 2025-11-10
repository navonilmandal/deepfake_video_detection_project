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

📈 Training Summary


Total training epochs: 10


Batch size: 16


Frames per video: 8


Achieved ~85–90% validation accuracy depending on dataset split


Model saved as best_model_resnet18.pth



🔍 Grad-CAM Explanation
To make the model explainable, Grad-CAM visualizations are used.
They highlight which regions of the face the model considers important while predicting.
This helps demonstrate why the model labels a frame as real or fake — a great addition for academic presentations.

🧪 Example Use Case


Upload a new, unknown video in Colab.


The model automatically:


Extracts a few frames.


Detects the face in each.


Predicts the probability of the face being fake.


Aggregates all frame scores for the final result.




Example:
Video: uploaded_video.mp4
Prediction: FAKE
Confidence: 83%


💾 Saving and Deployment
After training, the model is saved automatically as:
/content/deepfake_project/best_model_resnet18.pth

You can re-use this model later for real-time detection or deploy it in a web app using Flask/Streamlit.

👨‍💻 Author
Navonil Mandal
🎓BIT Mesra
📧 navonilmandal@gmail.com
🌐 [GitHub Profile](https://github.com/navonilmandal)

⭐ Acknowledgements


PyTorch for deep learning framework


OpenCV for image processing


Mendeley Dataset for providing real & fake videos


Google Colab for GPU-powered training environment



🏁 Final Note
This project demonstrates the use of deep learning and computer vision to detect AI-generated fake media.
It’s simple, explainable, and perfect for academic submissions or AI/ML portfolios.
⭐ If you find this useful, don’t forget to star the repo!

---

