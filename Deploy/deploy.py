# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from sklearn.metrics import confusion_matrix, classification_report
import os
import datetime
from tensorflow.keras.callbacks import TensorBoard

# Configure page
st.set_page_config(
    page_title="MNIST Digit Classifier 🎯",
    page_icon="🔢",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1 {color: #2a9d8f;}
    h2 {color: #264653;}
    .stProgress > div > div > div {background-color: #2a9d8f;}
    [data-testid="stSidebar"] {background-color: #e9f4f3;}
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("🧠 MNIST Digit Classification with CNN")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Parameters")
    epochs = st.slider("Number of epochs", 5, 30, 15)
    batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)
    st.markdown("---")
    st.markdown("**Model Architecture**")
    st.code("""
    Conv2D(32) -> BN -> MaxPool
    Conv2D(64) -> BN -> MaxPool
    Conv2D(128) -> BN -> MaxPool
    Dense(128) -> Dense(10)
    """)
    st.markdown("---")

# Load data
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Preprocessing
    x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test, y_train_cat, y_test_cat

x_train, y_train, x_test, y_test, y_train_cat, y_test_cat = load_data()

# Data visualization section
st.header("📊 Data Exploration")
cols = st.columns(4)
with cols[0]:
    st.metric("Training Samples", x_train.shape[0])
with cols[1]:
    st.metric("Test Samples", x_test.shape[0])
with cols[2]:
    st.metric("Image Size", f"{x_train.shape[1]}x{x_train.shape[2]}")
with cols[3]:
    st.metric("Number of Classes", 10)

st.subheader("Sample Training Images")
fig = plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
st.pyplot(fig)

# Model building
@st.cache_resource
def build_train_model(epochs=15, batch_size=64):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )

    # TensorBoard callback
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Training progress
    with st.spinner('🚀 Training model...'):
        history = model.fit(
            x_train, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[tensorboard_callback],
            verbose=1
        )
    st.success("✅ Training completed!")
    return model, history

# Train model
st.header("🛠 Model Training")
if st.button("Start/Restart Training"):
    model, history = build_train_model(epochs, batch_size)
    
    # Performance metrics
    st.header("📈 Performance Metrics")
    
    # Test evaluation
    st.subheader("🧪 Test Set Evaluation")
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test_cat, verbose=0)
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{accuracy:.2%}")
    cols[1].metric("Precision", f"{precision:.2%}")
    cols[2].metric("Recall", f"{recall:.2%}")
    cols[3].metric("Loss", f"{loss:.4f}")

    # Training curves
    st.subheader("📉 Learning Curves")
    fig = plt.figure(figsize=(16, 10))
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(history.history[metric], label=f"Train {metric}", marker='o')
        plt.plot(history.history[f"val_{metric}"], label=f"Val {metric}", linestyle='--', marker='x')
        plt.title(f"{metric.capitalize()} Curve")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    st.pyplot(fig)

    # Confusion matrix
    st.subheader("🤔 Confusion Matrix")
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='crest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Classification report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    st.table(report)

# Footer
st.markdown("---")
st.markdown("""
    ### 🎯 Key Features:
    - Convolutional Neural Network Architecture
    - Batch Normalization & Dropout Regularization
    - Real-time Training Metrics Tracking
    - Comprehensive Performance Analysis
    """)
st.markdown("---")
st.markdown("© **Copyright Shreyas Kasture**, All rights reserved")
