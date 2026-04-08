## COVID-19 X-ray Detection Web App
A deep learning-based web application built using FastAPI that predicts whether a chest X-ray image indicates COVID-19 infection or not.

# Features
- Easy X-ray image upload via web interface
- Deep Learning (CNN) based classification
- Multi-class prediction:
  COVID-19
  Viral Pneumonia
  Normal
- Real-time prediction with confidence score
- Camera integration support
- Fast and scalable backend using FastAPI


# Model Details
Model: CNN (Convolutional Neural Network)
Framework: TensorFlow / Keras
Input size: 150 x 150
Output: Predicted class (e.g., COVID / Normal / Pneumonia)


# Project Structure
project/
│
├── app.py
├── models/
│   ├── CNN_Covid19_Xray_Version.h5
│   └── Label_encoder.pkl
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── camera.html
│
├── uploads/
│
└── README.md
