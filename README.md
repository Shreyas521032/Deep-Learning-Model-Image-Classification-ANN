# 🧠 Deep Learning Models: Image Classification & Medical Risk Prediction

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

A repository containing two deep learning projects:
1. **MNIST Digit Classification** using CNNs
2. **Heart Disease Risk Prediction** using ANNs

---

## 📁 Project 1: MNIST Digit Classification (CNN)

### 🎯 Objective
Classify handwritten digits (0-9) from the MNIST dataset using a Convolutional Neural Network (CNN).

🌍 **Deployed Live Project**: [https://shreyas-deep-learning-model-image-classifiy.streamlit.app](https://shreyas-deep-learning-model-image-classifiy.streamlit.app/)

### 🏗️ Model Architecture
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### ✨ Key Features
- **Preprocessing:** Pixel normalization & one-hot encoding
- **Regularization:** Dropout layers and batch normalization
- **Evaluation:**
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
  - Training curves (accuracy/loss/precision/recall)
- **Visualization:** TensorBoard integration for real-time monitoring

---

## 📁 Project 2: Heart Disease Risk Prediction (ANN)

### 🎯 Objective
Predict heart disease risk from patient health records using an Artificial Neural Network (ANN).

🌍 **Deployed Live Project**: [https://shreyas-deep-learning-model-ann.streamlit.app](https://shreyas-deep-learning-model-ann.streamlit.app/)

### 🏗️ Model Architecture
```python
Sequential([
    Input(shape=(n_features,)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

### ✨ Key Features
- **Class Imbalance Handling:** `class_weight` parameter
- **Regularization:** Early stopping & dropout layers
- **Evaluation:**
  - ROC Curve & AUC Score
  - Precision-Recall Curve
  - Confusion Matrix
- **Optimization:** StandardScaler for feature normalization

---

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/Shreyas521032/Deep-Learning-Model-Image-Classification-ANN.git
cd Deep-Learning-Model-Image-Classification-ANN
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### For MNIST Classification:
```bash
python scenerio_based_case_study.py
```

Launch TensorBoard for training visualization:
```bash
tensorboard --logdir logs
```

### For Heart Disease Prediction:
```bash
python heart_disease_risk_prediction_using_deep_learning.py
```

---

## 📊 Results

### MNIST Model Performance
| Metric      | Score  |
|-------------|--------|
| Accuracy    | 99.2%  |
| Precision   | 99.3%  |
| Recall      | 99.1%  |

### Heart Disease Model Performance
| Metric      | Score |
|-------------|-------|
| AUC         | 0.92  |
| F1-Score    | 0.88  |

---

## 📈 Model Improvements

### For MNIST:
- Add data augmentation (rotation/translation)
- Implement transfer learning with pre-trained models
- Experiment with deeper architectures (ResNet blocks)

### For Heart Disease Prediction:
- Incorporate SMOTE for minority class oversampling
- Perform hyperparameter tuning with Keras Tuner
- Add feature importance analysis

---

## License

© 2025 Shreyas Kasture

All rights reserved.

This software and its source code are the intellectual property of the author. Unauthorized copying, distribution, modification, or usage in any form is strictly prohibited without explicit written permission.

For licensing inquiries, please contact: shreyas200410@gmail.com
