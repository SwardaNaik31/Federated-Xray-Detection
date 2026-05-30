# 🩺 Federated Learning Based X-Ray Disease Detection System

## 📌 Project Overview

This project implements a **Federated Learning-based X-Ray Disease Detection System** using Deep Learning and the Flower framework. The system enables multiple clients (hospitals or medical centers) to collaboratively train an AI model without sharing their sensitive patient data.

Instead of transferring medical images to a central server, each client trains the model locally and shares only model parameters. The server aggregates these updates to create a global model, ensuring privacy, security, and efficient collaborative learning.

---

## 🎯 Objectives

- Detect diseases from X-ray images using Deep Learning.
- Preserve patient privacy through Federated Learning.
- Enable collaborative model training without sharing raw medical data.
- Improve model performance using distributed datasets.
- Build a user-friendly web interface for disease prediction.

---

## 🚀 Key Features

- Privacy-preserving Federated Learning
- CNN-based X-ray image classification
- Distributed client training
- Flower-based federated server
- Secure model parameter sharing
- Real-time disease prediction
- Web-based user interface

---

## 🛠️ Technologies Used

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)

### Federated Learning
- Flower Framework

### Libraries
- NumPy
- Pandas
- OpenCV
- Matplotlib

---

## 🏗️ System Architecture

```text
                 X-Ray Dataset
                        │
                        ▼
                Dataset Splitting
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    Client 1        Client 2        Client 3
 Local Training   Local Training   Local Training
        │               │               │
        └───────────────┼───────────────┘
                        ▼
               Federated Server
              (Weight Aggregation)
                        │
                        ▼
                 Global CNN Model
                        │
                        ▼
                Disease Prediction
```

---

## 📂 Project Structure

```text
PBL-Project/
│
├── client/
│   ├── client1.py (i.e. Hospital_A)
│   ├── client2.py (i.e. Hospital_B)
│   └── client3.py (i.e. Hospital_C)
│
├── server/
│   └── server.py
│
├── frontend/
│   ├── templates/
│   ├── static/
│   └── app.py
│
├── dataset/
│
├── split_dataset.py
├── model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

### Step 1: Dataset Preparation
- Collect X-ray images.
- Split the dataset among multiple clients.

### Step 2: Local Training
- Each client trains the CNN model on its local dataset.
- Data remains on the client device.

### Step 3: Parameter Sharing
- Clients send model weights to the federated server.

### Step 4: Federated Aggregation
- Server aggregates weights received from all clients.
- A new global model is generated.

### Step 5: Prediction
- Users upload X-ray images through the web interface.
- The trained global model predicts disease status.
<img width="1328" height="686" alt="ss1" src="https://github.com/user-attachments/assets/0ff4cbd2-f1bf-4930-abb4-d082e4bed733" />

---

## 🧠 Deep Learning Model

The project uses a **Convolutional Neural Network (CNN)** for image classification.

### CNN Components
- Convolution Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Softmax Output Layer

---

## ▶️ Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/SwardaNaik31/Federated-Xray-Detection.git
cd xray-federated-learning
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Split Dataset

```bash
python split_dataset.py
```

### Start Federated Server

```bash
python server.py
```

### Start Clients (Separate Terminals)

```bash
python client1.py
```

```bash
python client2.py
```

```bash
python client3.py
```

### Run Web Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📊 Results

The model provides:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Disease Prediction

### Example Output

```text
Prediction: Disease Detected
Confidence: 96.3%
```

---

## 🔒 Advantages

- Patient data never leaves local devices.
- Enhanced privacy and security.
- Reduced risk of data leakage.
- Scalable distributed learning.
- Better model generalization.

---

## 🌍 Applications

### Healthcare
- Pneumonia Detection
- Lung Disease Detection
- Medical Image Classification

### Other Domains
- Banking Fraud Detection
- Autonomous Vehicles
- Cybersecurity
- IoT Networks
- Recommendation Systems

---

## 🔮 Future Scope

- Multi-disease classification
- Mobile application deployment
- Cloud-based federated learning
- Explainable AI integration
- Real hospital deployment

---

## 👩‍💻 Authors

**Swarda Naik**

Academic Project – Federated Learning Based X-Ray Disease Detection System.

---

<img width="1328" height="686" alt="ss1" src="https://github.com/user-attachments/assets/f3ff4fed-7b38-4ed9-9d67-eae712492ae0" />
<img width="1328" height="686" alt="ss1" src="https://github.com/user-attachments/assets/5672a737-fdc2-40b5-8010-239f76383493" />
<img width="1328" height="686" alt="ss1" src="https://github.com/user-attachments/assets/9eea7e44-d383-40ab-9dac-8408316bc24c" />
<img width="1328" height="686" alt="ss1" src="https://github.com/user-attachments/assets/51e83573-b1d1-45f5-ab3b-474ef84d58de" />
