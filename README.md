# Smart-Poultry-Disease-Identifier
Smart Poultry Disease Identifier using MobileNetV2 – A Deep Learning based web application for real-time poultry disease detection from image inputs.

Smart Poultry Disease Identifier using MobileNetV2 – A Deep Learning based web application for real-time poultry disease detection from image inputs.



🐔 Smart Poultry Disease Identifier
🚀 Deep Learning Based Poultry Health Monitoring System (MobileNetV2)
<p align="center"> <img src="https://img.shields.io/badge/Deep%20Learning-MobileNetV2-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge"/> <img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge"/> <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/> </p>
📌 Project Overview

The Smart Poultry Disease Identifier is a deep learning–based image classification system designed to automatically detect poultry diseases from image inputs.

The system leverages Transfer Learning using MobileNetV2 to classify poultry images as:

✅ Healthy

❌ Diseased

It provides real-time prediction along with a confidence score through a user-friendly web interface built using Streamlit.

🧠 Model Architecture

📌 Base Model: MobileNetV2

📌 Transfer Learning Applied

📌 Input Size: 224 × 224 × 3

📌 Lightweight & Efficient

📌 Suitable for Real-Time Deployment

MobileNetV2 uses:

🔹Depthwise Separable Convolutions

🔹Inverted Residual Blocks

🔹Linear Bottlenecks

🔹This ensures high accuracy with low computational cost.

🔄 System Workflow
Image Upload → Preprocessing → Model Prediction → Decision Layer → Output Display

1️⃣ User uploads poultry image
2️⃣ Image is resized & normalized
3️⃣ MobileNetV2 extracts features
4️⃣ Highest probability class selected
5️⃣ Result + Confidence Score displayed

📊 Performance Evaluation

The model performance is evaluated using:

✔ Accuracy

✔ Precision

✔ Recall

✔ F1-Score

✔ Confusion Matrix

These metrics ensure reliability and robustness of disease detection.

🛠️ Technologies Used
🔹Category	Tools

🔹Programming	Python

🔹Deep Learning	TensorFlow / Keras

🔹Computer Vision	OpenCV

🔹UI	Streamlit

🔹Data Processing	NumPy

🔹Evaluation	Scikit-learn


📂 Project Structure
Smart-Poultry-Disease-Identifier/
│
├── app.py
├── requirements.txt
├── mobilenetv2.h5
├── dataset/
├── notebooks/
└── README.md

⚙️ Installation (Local Setup)

pip install -r requirements.txt
streamlit run app.py

Then open:

http://localhost:8501

🌍 Deployment

The application can be deployed using:

🔹 Streamlit Cloud

🔹 HuggingFace Spaces

🔹 Render

🔹 AWS / GCP

🎯 Objectives

🔹Early disease detection

🔹Reduce poultry mortality

🔹Minimize economic losses

🔹Enable smart farming

🔹Provide cost-effective AI solution

🚀 Future Enhancements

🔹Multi-disease classification

🔹IoT-based smart farm integration

🔹Mobile application version

🔹TensorFlow Lite deployment

🔹Automated treatment recommendation

👨‍💻 Author

Karthick S
B.Sc Artificial Intelligence & Machine Learning
Final Year Project

⭐ If you like this project

Give it a ⭐ on GitHub!
