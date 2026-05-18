# Federated Learning Based X-Ray Pneumonia Detection System

This project is an AI-based healthcare application that detects pneumonia from chest X-ray images using Deep Learning and Federated Learning.

The system allows multiple clients to train the model locally without sharing patient data, improving privacy and security.

---

## Features

- Pneumonia Detection using CNN
- Federated Learning using Flower Framework
- Privacy Preserving Training
- Flask Web Application
- Chest X-Ray Image Prediction
- Model Accuracy Visualization

---

## Technologies Used

- Python
- TensorFlow / Keras
- Flask
- Flower (FLWR)
- NumPy
- OpenCV
- HTML/CSS

---

## Project Structure

```bash
PBL-project/
│
├── server/
│ └── server.py
│
├── client/
│ └── client.py
│
├── templates/
│ └── index.html
│
├── static/
│
├── dataset/
│
├── model.py
├── split_dataset.py
├── requirements.txt
└── README.md
