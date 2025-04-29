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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard
import random
import time

# Configure page
st.set_page_config(
    page_title="Deep Learning Playground 🧠",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with more vibrant design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e6eef8 100%);
        border-radius: 20px;
        padding: 20px;
    }

    h1 {
        background: linear-gradient(90deg, #4e54c8, #8f94fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -1px;
    }

    h2 {
        color: #4e54c8;
        font-weight: 600;
        border-bottom: 2px solid #8f94fb;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    h3 {
        color: #6c63ff;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4e54c8, #8f94fb);
        color: white;
        border-radius: 50px;
        padding: 10px 25px;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(78, 84, 200, 0.3);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 20px rgba(78, 84, 200, 0.5);
    }

    [data-testid="stSidebar"] {
        background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: white !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.8) !important;
    }

    .highlight-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        transition: transform 0.3s ease;
    }

    .highlight-card:hover {
        transform: translateY(-5px);
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4e54c8, #8f94fb);
        animation: pulse 2s infinite;
    }

    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
        background: linear-gradient(90deg, #4e54c8, #8f94fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 1rem;
        color: #555;
        font-weight: 500;
    }

    .label-pill {
        background: linear-gradient(90deg, #4e54c8, #8f94fb);
        color: white;
        padding: 5px 15px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 10px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(78, 84, 200, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(78, 84, 200, 0); }
        100% { box-shadow: 0 0 0 0 rgba(78, 84, 200, 0); }
    }

    .footer {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        margin-top: 50px;
    }

    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        border: none;
        padding: 10px 20px;
        color: #4e54c8;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4e54c8 !important;
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)

# Custom header with animated title effect
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 3.5rem; margin-bottom: 0;">🧠 Deep Learning Playground</h1>
    <p style="font-size: 1.5rem; color: #666; margin-top: 5px;">✨ Interactive MNIST Digit Classification with Deep Learning ✨</p>
</div>
""", unsafe_allow_html=True)


# Fun loading animation for first-time visitors
if 'first_load' not in st.session_state:
    st.session_state.first_load = True
    placeholder = st.empty()
    with placeholder.container():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = ["🤖 Initializing neural pathways...", 
                "🧩 Assembling neural architecture...", 
                "🔮 Calibrating prediction matrix...", 
                "🚀 Warming up GPUs...",
                "🧪 Running final diagnostics..."]
        
        for i, step in enumerate(steps):
            status_text.markdown(f"<h3 style='text-align:center'>{step}</h3>", unsafe_allow_html=True)
            for j in range(20):
                progress_bar.progress((i * 20 + j + 1) / 100)
                time.sleep(0.02)
    
    placeholder.empty()
    st.session_state.first_load = False

# Animated sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align:center; margin-bottom:30px;'>🎛️ Control Panel</h2>", unsafe_allow_html=True)
    
    # Add some fun neural network ASCII art
    st.markdown("""
    ```
    Input  →  Input  →  Input
      ↓        ↓        ↓
    Hidden → Hidden → Hidden
      ↓        ↓        ↓
    Output → Output → Output
    ```
""")
    # Emoji-rich training parameters
    st.markdown("<div class='label-pill'>🔄 Training Parameters</div>", unsafe_allow_html=True)
    epochs = st.slider("🔁 Number of epochs", 5, 30, 15)
    batch_size = st.select_slider("📦 Batch size", options=[32, 64, 128, 256], value=64)
    
    # Model complexity section
    st.markdown("<div class='label-pill'>🧬 Model Complexity</div>", unsafe_allow_html=True)
    model_complexity = st.radio("", ["🔍 Simple", "⚖️ Balanced", "🔬 Complex"], index=1, horizontal=True)
    
    # Learning parameters
    st.markdown("<div class='label-pill'>📊 Learning Parameters</div>", unsafe_allow_html=True)
    learning_rate = st.select_slider("🎯 Learning rate", 
                                   options=[0.0001, 0.001, 0.01, 0.1], 
                                   value=0.001,
                                   format_func=lambda x: f"{x:.4f}")
    
    dropout_rate = st.slider("🪂 Dropout rate", 0.1, 0.5, 0.3, 0.05)

    # Fun "Did you know" facts
    st.markdown("---")
    ml_facts = [
        "Neural networks were first proposed in 1943! 🧠👨‍🔬",
        "MNIST has 70,000 images of handwritten digits! ✍️",
        "CNNs were inspired by the visual cortex! 👁️",
        "Adding more layers doesn't always mean better results! 📊"
    ]
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.1); padding:15px; border-radius:10px; margin-top:30px'>
        <h3 style='color:white; font-size:16px'>💡 Did you know?</h3>
        <p style='color:white; font-size:14px'>{random.choice(ml_facts)}</p>
        <p style='margin:0; font-size:12px; opacity:0.8;'>— Developed by Shreyas Kasture</p>
    </div>
    """, unsafe_allow_html=True)

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

# Main content area with tabs
tabs = st.tabs(["🔍 Data Explorer", "🧠 Neural Network", "📊 Results & Analysis", "🎯 Prediction Playground"])

# Tab 1: Data Explorer
with tabs[0]:
    st.markdown("<h2>🔍 Exploring the MNIST Dataset</h2>", unsafe_allow_html=True)
    
    # Interactive dataset metrics in custom cards
    st.markdown("<div style='display:flex; gap:20px;'>", unsafe_allow_html=True)
    
    metrics = [
        {"label": "Training Samples", "value": f"{x_train.shape[0]:,}", "icon": "📚"},
        {"label": "Test Samples", "value": f"{x_test.shape[0]:,}", "icon": "🧪"},
        {"label": "Image Size", "value": f"{x_train.shape[1]}x{x_train.shape[2]}", "icon": "📏"},
        {"label": "Number of Classes", "value": "10", "icon": "🔢"}
    ]
    
    cols = st.columns(4)
    for i, col in enumerate(cols):
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">{metrics[i]['icon']}</div>
            <div class="metric-value">{metrics[i]['value']}</div>
            <div class="metric-label">{metrics[i]['label']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3>✨ Dataset Distribution</h3>", unsafe_allow_html=True)
    
    # Class distribution visualization with Plotly
    digit_counts = np.bincount(y_train)
    fig = px.bar(
        x=list(range(10)), 
        y=digit_counts,
        labels={'x': 'Digit', 'y': 'Frequency'},
        color=digit_counts,
        color_continuous_scale='viridis',
        title="Distribution of Digits in Training Set"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive sample explorer
    st.markdown("<h3>👀 Sample Images Explorer</h3>", unsafe_allow_html=True)
    
    sample_cols = st.columns([3, 1])
    with sample_cols[1]:
        st.markdown("<div style='background:white; padding:20px; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        digit_to_show = st.radio("🔢 Select a digit", range(10))
        num_samples = st.slider("📑 Samples to show", 8, 24, 16, 4)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with sample_cols[0]:
        # Get indices for the selected digit
        digit_indices = np.where(y_train == digit_to_show)[0]
        
        # Randomly sample from those indices
        sample_indices = np.random.choice(digit_indices, min(num_samples, len(digit_indices)), replace=False)
        
        fig = plt.figure(figsize=(10, 10))
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        for i, idx in enumerate(sample_indices):
            if i < num_samples:
                plt.subplot(rows, cols, i + 1)
                plt.imshow(x_train[idx].reshape(28, 28), cmap='viridis')
                plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Image transformation explorer
    st.markdown("<h3>🔄 Image Transformations</h3>", unsafe_allow_html=True)
    
    transform_cols = st.columns(2)
    with transform_cols[0]:
        st.markdown("<div style='background:white; padding:20px; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        random_idx = st.button("🎲 Pick Random Image", key="random_transform")
        if random_idx or 'transform_idx' not in st.session_state:
            st.session_state.transform_idx = np.random.randint(0, len(x_train))
        
        # Original image
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(x_train[st.session_state.transform_idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Original Image (Digit: {y_train[st.session_state.transform_idx]})")
        ax.axis('off')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with transform_cols[1]:
        st.markdown("<div style='background:white; padding:20px; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        transform_type = st.selectbox("🔄 Select Transformation", 
                                     ["Invert", "Rotate", "Add Noise", "Blur"])
        
        # Apply transformation
        img = x_train[st.session_state.transform_idx].reshape(28, 28).copy()
        
        if transform_type == "Invert":
            img = 1 - img
            title = "Inverted Image"
        elif transform_type == "Rotate":
            angle = st.slider("🔄 Rotation Angle", -180, 180, 45)
            from scipy.ndimage import rotate
            img = rotate(img, angle, reshape=False)
            title = f"Rotated Image ({angle}°)"
        elif transform_type == "Add Noise":
            noise_level = st.slider("🔊 Noise Level", 0.0, 0.5, 0.1, 0.05)
            noise = np.random.normal(0, noise_level, img.shape)
            img = np.clip(img + noise, 0, 1)
            title = f"Noisy Image ({noise_level:.2f})"
        elif transform_type == "Blur":
            blur_level = st.slider("🌫️ Blur Level", 1, 5, 2)
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, sigma=blur_level/2)
            title = f"Blurred Image (σ={blur_level/2:.1f})"
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Neural Network
with tabs[1]:
    st.markdown("<h2>🧠 Neural Network Architecture</h2>", unsafe_allow_html=True)
    
    # Define model based on complexity choice
    def get_model_architecture(complexity):
        if complexity == "🔍 Simple":
            return [
                ["Conv2D", 16, "(3, 3)", "relu", "(28, 28, 1)"],
                ["MaxPooling2D", "(2, 2)"],
                ["Flatten"],
                ["Dense", 64, "relu"],
                ["Dense", 10, "softmax"]
            ]
        elif complexity == "⚖️ Balanced":
            return [
                ["Conv2D", 32, "(3, 3)", "relu", "(28, 28, 1)"],
                ["BatchNormalization"],
                ["MaxPooling2D", "(2, 2)"],
                ["Dropout", dropout_rate],
                
                ["Conv2D", 64, "(3, 3)", "relu"],
                ["BatchNormalization"],
                ["MaxPooling2D", "(2, 2)"],
                ["Dropout", dropout_rate],
                
                ["Conv2D", 128, "(3, 3)", "relu"],
                ["BatchNormalization"],
                ["MaxPooling2D", "(2, 2)"],
                ["Dropout", dropout_rate],
                
                ["Flatten"],
                ["Dense", 128, "relu"],
                ["Dropout", dropout_rate],
                ["Dense", 10, "softmax"]
            ]
        else:  # Complex
            return [
                ["Conv2D", 32, "(3, 3)", "relu", "(28, 28, 1)"],
                ["BatchNormalization"],
                ["Conv2D", 32, "(3, 3)", "relu"],
                ["BatchNormalization"],
                ["MaxPooling2D", "(2, 2)"],
                ["Dropout", dropout_rate],
                
                ["Conv2D", 64, "(3, 3)", "relu"],
                ["BatchNormalization"],
                ["Conv2D", 64, "(3, 3)", "relu"],
                ["BatchNormalization"],
                ["MaxPooling2D", "(2, 2)"],
                ["Dropout", dropout_rate],
                
                ["Conv2D", 128, "(3, 3)", "relu"],
                ["BatchNormalization"],
                ["Conv2D", 128, "(3, 3)", "relu"],
                ["BatchNormalization"],
                ["MaxPooling2D", "(2, 2)"],
                ["Dropout", dropout_rate],
                
                ["Flatten"],
                ["Dense", 256, "relu"],
                ["BatchNormalization"],
                ["Dropout", dropout_rate],
                ["Dense", 128, "relu"],
                ["Dropout", dropout_rate],
                ["Dense", 10, "softmax"]
            ]
    
    architecture = get_model_architecture(model_complexity)
    
    # Visualize architecture
    cols = st.columns([2, 1])
    
    with cols[1]:
        st.markdown("<div style='background:white; padding:20px; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        st.markdown("<h3>📋 Layer Summary</h3>", unsafe_allow_html=True)
        
        # Generate layer summary 
        for i, layer in enumerate(architecture):
            layer_type = layer[0]
            
            if layer_type == "Conv2D":
                st.markdown(f"**{i+1}. {layer_type}** - {layer[1]} filters, {layer[2]} kernel")
            elif layer_type == "Dense":
                st.markdown(f"**{i+1}. {layer_type}** - {layer[1]} units, {layer[2]} activation")
            elif layer_type == "Dropout":
                st.markdown(f"**{i+1}. {layer_type}** - rate: {layer[1]}")
            else:
                st.markdown(f"**{i+1}. {layer_type}**" + (f" - {layer[1]}" if len(layer) > 1 else ""))
        
        st.markdown("<h3>⚙️ Training Config</h3>", unsafe_allow_html=True)
        st.code(f"""model.compile(
    optimizer=Adam(lr={learning_rate}),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)""")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with cols[0]:
        # Create a visual representation of the network
        st.markdown("<h3>🧩 Network Visualization</h3>", unsafe_allow_html=True)
        
        # Generate network diagram
        fig = go.Figure()
        
        # Define node positions
        input_layer = {"x": [0], "y": [i for i in range(-3, 4)], "name": ["Input"]}
        
        # Count how many Conv/Pool/BN groups we have
        conv_blocks = 0
        dense_blocks = 0
        
        for layer in architecture:
            if layer[0] == "Conv2D":
                conv_blocks += 1
            elif layer[0] == "Dense" and layer[0] != "Dense" and layer[1] != 10:
                dense_blocks += 1
        
        # Create hidden layers based on architecture complexity
        hidden_layers = []
        x_position = 1
        current_block = 0
        dense_seen = False
        
        for layer in architecture:
            layer_type = layer[0]
            
            if layer_type == "Conv2D":
                current_block += 1
                hidden_layers.append({
                    "x": [x_position] * 5,
                    "y": [i for i in range(-2, 3)],
                    "name": [f"Conv {current_block}"] * 5,
                    "color": "#4e54c8",
                    "size": 10
                })
                x_position += 1
            elif layer_type == "MaxPooling2D":
                hidden_layers.append({
                    "x": [x_position] * 3,
                    "y": [i for i in range(-1, 2)],
                    "name": [f"Pool {current_block}"] * 3,
                    "color": "#8f94fb", 
                    "size": 8
                })
                x_position += 1
            elif layer_type == "Flatten":
                hidden_layers.append({
                    "x": [x_position] * 3,
                    "y": [i for i in range(-1, 2)],
                    "name": ["Flatten"] * 3,
                    "color": "#764ba2",
                    "size": 8
                })
                x_position += 1
                dense_seen = True
            elif layer_type == "Dense" and layer[1] != 10:
                if dense_seen:
                    hidden_layers.append({
                        "x": [x_position] * 4,
                        "y": [i for i in range(-2, 2)],
                        "name": [f"Dense {layer[1]}"] * 4,
                        "color": "#667eea",
                        "size": 10
                    })
                    x_position += 1
        
        output_layer = {"x": [x_position], "y": [i for i in range(-5, 5)], "name": ["Output"] * 10}
        
        # Add edges (connections between layers)
        edges_x = []
        edges_y = []
        
        # Add connections from input to first hidden
        for i_in in range(len(input_layer["y"])):
            for i_h in range(len(hidden_layers[0]["y"])):
                edges_x.append(input_layer["x"][0])
                edges_x.append(hidden_layers[0]["x"][0])
                edges_x.append(None)
                edges_y.append(input_layer["y"][i_in])
                edges_y.append(hidden_layers[0]["y"][i_h])
                edges_y.append(None)
        
        # Add connections between hidden layers
        for h in range(len(hidden_layers) - 1):
            for i_h1 in range(len(hidden_layers[h]["y"])):
                for i_h2 in range(len(hidden_layers[h + 1]["y"])):
                    edges_x.append(hidden_layers[h]["x"][0])
                    edges_x.append(hidden_layers[h + 1]["x"][0])
                    edges_x.append(None)
                    edges_y.append(hidden_layers[h]["y"][i_h1])
                    edges_y.append(hidden_layers[h + 1]["y"][i_h2])
                    edges_y.append(None)
        
        # Add connections from last hidden to output
        for i_h in range(len(hidden_layers[-1]["y"])):
            for i_out in range(len(output_layer["y"])):
                edges_x.append(hidden_layers[-1]["x"][0])
                edges_x.append(output_layer["x"][0])
                edges_x.append(None)
                edges_y.append(hidden_layers[-1]["y"][i_h])
                edges_y.append(output_layer["y"][i_out])
                edges_y.append(None)
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edges_x,
            y=edges_y,
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.2)', width=1),
            hoverinfo='none'
        ))
        
        # Add nodes for each layer
        # Input layer
        fig.add_trace(go.Scatter(
            x=input_layer["x"] * len(input_layer["y"]),
            y=input_layer["y"],
            mode='markers+text',
            marker=dict(size=12, color="#764ba2", line=dict(width=1, color="#ffffff")),
            text=input_layer["name"],
            textposition="middle left",
            hoverinfo='text',
            hovertext=[f"28x28 Input Image"] * len(input_layer["y"])
        ))
        
        # Hidden layers
        for i, layer in enumerate(hidden_layers):
            fig.add_trace(go.Scatter(
                x=layer["x"],
                y=layer["y"],
                mode='markers',
                marker=dict(
                    size=layer.get("size", 10), 
                    color=layer.get("color", "#4e54c8"), 
                    line=dict(width=1, color="#ffffff")
                ),
                hoverinfo='text',
                hovertext=[f"{layer['name'][0]}"] * len(layer["y"])
            ))
        
        # Output layer
        fig.add_trace(go.Scatter(
            x=output_layer["x"] * len(output_layer["y"]),
            y=output_layer["y"],
            mode='markers+text',
            marker=dict(size=12, color="#4e54c8", line=dict(width=1, color="#ffffff")),
            text=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            textposition="middle right",
            hoverinfo='text',
            hovertext=[f"Output - Digit {i}" for i in range(10)]
        ))
        
        # Update layout
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Train model button
    st.markdown("<h3>🚀 Train Your Neural Network</h3>", unsafe_allow_html=True)
    train_col1, train_col2 = st.columns([3, 1])
    
    with train_col2:
        st.markdown("<div style='background:white; padding:20px; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        train_fraction = st.slider("📊 Training data fraction", 0.1, 1.0, 0.2, 0.1)
        validation_split = st.slider("🔀 Validation split", 0.1, 0.3, 0.2, 0.05)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with train_col1:
        train_button = st.button("🚀 Train Model", key="train_model")
        
        if train_button or ('model' in st.session_state and 'history' in st.session_state):
            with st.spinner("🧠 Training in progress..."):
                # Build model based on selected complexity
                if 'model' not in st.session_state:
                    model = Sequential()
                    
                    # Add layers based on architecture
                    for layer in architecture:
                        if layer[0] == "Conv2D":
                            if len(layer) > 4:  # Has input shape
                                model.add(Conv2D(layer[1], eval(layer[2]), activation=layer[3], input_shape=eval(layer[4])))
                            else:
                                model.add(Conv2D(layer[1], eval(layer[2]), activation=layer[3]))
                        elif layer[0] == "MaxPooling2D":
                            model.add(MaxPooling2D(eval(layer[1])))
                        elif layer[0] == "Flatten":
                            model.add(Flatten())
                        elif layer[0] == "Dense":
                            model.add(Dense(layer[1], activation=layer[2]))
                        elif layer[0] == "Dropout":
                            model.add(Dropout(layer[1]))
                        elif layer[0] == "BatchNormalization":
                            model.add(BatchNormalization())
                    
                    # Compile model
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Save to session state
                    st.session_state.model = model
                
                # Train model
                if 'history' not in st.session_state:
                    # Use only a fraction of training data for demonstration
                    train_size = int(x_train.shape[0] * train_fraction)
                    x_train_subset = x_train[:train_size]
                    y_train_subset = y_train_cat[:train_size]
                    
                    # Add TensorBoard callback
                    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
                    
                    # Train the model
                    history = st.session_state.model.fit(
                        x_train_subset, y_train_subset,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=1,
                        callbacks=[tensorboard_callback]
                    )
                    
                    # Save history to session state
                    st.session_state.history = {
                        'accuracy': history.history['accuracy'],
                        'val_accuracy': history.history['val_accuracy'],
                        'loss': history.history['loss'],
                        'val_loss': history.history['val_loss']
                    }
                    
                # Display training progress
                if 'history' in st.session_state:
                    # Create subplot with 2 rows and 1 column
                    fig = make_subplots(rows=2, cols=1, subplot_titles=("Model Accuracy", "Model Loss"))
                    
                    # Add accuracy traces
                    fig.add_trace(
                        go.Scatter(y=st.session_state.history['accuracy'], 
                                  name="Training Accuracy",
                                  line=dict(color="#4e54c8", width=3)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(y=st.session_state.history['val_accuracy'], 
                                  name="Validation Accuracy",
                                  line=dict(color="#8f94fb", width=3, dash='dash')),
                        row=1, col=1
                    )
                    
                    # Add loss traces
                    fig.add_trace(
                        go.Scatter(y=st.session_state.history['loss'], 
                                  name="Training Loss",
                                  line=dict(color="#764ba2", width=3)),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(y=st.session_state.history['val_loss'], 
                                  name="Validation Loss",
                                  line=dict(color="#667eea", width=3, dash='dash')),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    fig.update_xaxes(title_text="Epoch", showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
                    
                    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Results & Analysis
with tabs[2]:
    st.markdown("<h2>📊 Results & Model Analysis</h2>", unsafe_allow_html=True)
    
    if 'model' not in st.session_state or 'history' not in st.session_state:
        st.info("👉 Please train your model in the Neural Network tab first!")
    else:
        # Model evaluation on test set
        st.markdown("<h3>🎯 Model Evaluation</h3>", unsafe_allow_html=True)
        
        # Evaluate button
        evaluate_button = st.button("📈 Evaluate Model on Test Data", key="evaluate_model")
        
        if evaluate_button or 'evaluation' in st.session_state:
            with st.spinner("Evaluating model on test data..."):
                if 'evaluation' not in st.session_state:
                    # Evaluate model on test data
                    results = st.session_state.model.evaluate(x_test, y_test_cat, verbose=0)
                    y_pred = st.session_state.model.predict(x_test)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    
                    # Calculate confusion matrix
                    cm = confusion_matrix(y_test, y_pred_classes)
                    
                    # Classification report
                    report = classification_report(y_test, y_pred_classes, output_dict=True)
                    
                    # Store results in session state
                    st.session_state.evaluation = {
                        'results': results,
                        'confusion_matrix': cm,
                        'report': report,
                        'y_pred': y_pred,
                        'y_pred_classes': y_pred_classes
                    }
                
                # Display evaluation metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:2rem;">🎯</div>
                        <div class="metric-value">{st.session_state.evaluation['results'][1]*100:.2f}%</div>
                        <div class="metric-label">Test Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Calculate average precision from classification report
                    avg_precision = st.session_state.evaluation['report']['weighted avg']['precision']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:2rem;">📏</div>
                        <div class="metric-value">{avg_precision*100:.2f}%</div>
                        <div class="metric-label">Avg. Precision</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Calculate average recall from classification report
                    avg_recall = st.session_state.evaluation['report']['weighted avg']['recall']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:2rem;">🔍</div>
                        <div class="metric-value">{avg_recall*100:.2f}%</div>
                        <div class="metric-label">Avg. Recall</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confusion Matrix
                st.markdown("<h3>🧩 Confusion Matrix</h3>", unsafe_allow_html=True)

                fig = px.imshow(
    st.session_state.evaluation['confusion_matrix'],
    labels=dict(x="Predicted Digit", y="True Digit", color="Count"),
    x=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    y=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    color_continuous_scale="Viridis",
    aspect="equal"
                 )

                fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=20, b=20),
    height=500,
    coloraxis_colorbar=dict(
        title="Prediction<br>Count",  # Custom color bar title
        tickfont=dict(color='white'),  # Change tick label color
        titlefont=dict(color='white')  # Change title color
    )

                st.plotly_chart(fig, use_container_width=True)


                # Per-class metrics
                st.markdown("<h3>📊 Per-Class Performance</h3>", unsafe_allow_html=True)
                
                # Create a DataFrame from classification report
                per_class_metrics = []
                for i in range(10):
                    per_class_metrics.append({
                        'Digit': str(i),
                        'Precision': st.session_state.evaluation['report'][str(i)]['precision'],
                        'Recall': st.session_state.evaluation['report'][str(i)]['recall'],
                        'F1-Score': st.session_state.evaluation['report'][str(i)]['f1-score'],
                        'Support': st.session_state.evaluation['report'][str(i)]['support']
                    })
                
                df_metrics = pd.DataFrame(per_class_metrics)
                
                # Create radar chart for each digit
                st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;'>", unsafe_allow_html=True)
                for i in range(10):
                    digit_metrics = df_metrics[df_metrics['Digit'] == str(i)].iloc[0]
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[digit_metrics['Precision'], digit_metrics['Recall'], digit_metrics['F1-Score'], 
                           digit_metrics['Support']/1000],  # Normalize support
                        theta=['Precision', 'Recall', 'F1-Score', 'Support'],
                        fill='toself',
                        name=f'Digit {i}',
                        line_color=px.colors.sequential.Viridis[i % 10]
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=250,
                        width=250
                    )
                    fig.update_layout(title=f"Digit {i}")
                    
                    st.plotly_chart(fig, use_container_width=False)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Misclassified examples
                st.markdown("<h3>❌ Misclassified Examples</h3>", unsafe_allow_html=True)
                
                # Find misclassified examples
                misclassified_indices = np.where(y_test != st.session_state.evaluation['y_pred_classes'])[0]
                
                if len(misclassified_indices) > 0:
                    # Show random misclassified examples
                    num_examples = min(8, len(misclassified_indices))
                    random_misclassified = np.random.choice(misclassified_indices, num_examples, replace=False)
                    
                    cols = st.columns(4)
                    for i, idx in enumerate(random_misclassified):
                        col = cols[i % 4]
                        with col:
                            st.markdown(f"""
                            <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 6px 15px rgba(0,0,0,0.1); margin-bottom: 20px;">
                                <div style="text-align: center; font-weight: bold; color: #4e54c8;">
                                    True: {y_test[idx]} | Predicted: {st.session_state.evaluation['y_pred_classes'][idx]}
                                </div>
                            """, unsafe_allow_html=True)
                            
                            fig, ax = plt.subplots(figsize=(3, 3))
                            ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
                            ax.axis('off')
                            st.pyplot(fig)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.success("Congratulations! Your model classified all test examples correctly!")

# Tab 4: Prediction Playground
with tabs[3]:
    st.markdown("<h2>🎯 Prediction Playground</h2>", unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.info("👉 Please train your model in the Neural Network tab first!")
    else:
        st.markdown("<h3>✏️ Draw a Digit</h3>", unsafe_allow_html=True)
        
        # Create two columns
        draw_cols = st.columns([2, 1])
        
        with draw_cols[0]:
            # Use st.empty() for updating canvas content
            canvas_placeholder = st.empty()
            
            # Initialize canvas or use existing from session
            if 'canvas_image' not in st.session_state:
                # Create a white canvas
                canvas_image = np.ones((280, 280), dtype=np.float32)
                st.session_state.canvas_image = canvas_image
            
            # Display current canvas
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(st.session_state.canvas_image, cmap='gray')
            ax.axis('off')
            canvas_placeholder.pyplot(fig)
            
            # Drawing controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧹 Clear", key="clear_canvas"):
                    st.session_state.canvas_image = np.ones((280, 280), dtype=np.float32)
                    st.rerun()
            
            with col2:
                if st.button("🎲 Random Digit", key="random_digit"):
                    # Pick a random digit from test set
                    idx = np.random.randint(0, len(x_test))
                    # Resize to 280x280 for better display
                    from scipy.ndimage import zoom
                    img = zoom(x_test[idx].reshape(28, 28), 10, order=0)
                    st.session_state.canvas_image = img
                    # Also store the true label
                    st.session_state.canvas_true_label = y_test[idx]
                    st.rerun()
            
            with col3:
                if st.button("🔄 Predict", key="predict_button"):
                    # Downscale image to 28x28
                    from skimage.transform import resize
                    small_img = resize(st.session_state.canvas_image, (28, 28))
                    
                    # Normalize and reshape for model
                    img_for_model = small_img.reshape(1, 28, 28, 1)
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(img_for_model)
                    predicted_class = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_class]
                    
                    # Store prediction results
                    st.session_state.prediction = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'all_probs': prediction[0]
                    }
        
        with draw_cols[1]:
            st.markdown("<div style='background:white; padding:20px; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
            st.markdown("<h3>🎨 Drawing Tools</h3>", unsafe_allow_html=True)
            
            # Drawing simulation
            st.info("In this demo version, use the 'Random Digit' button to load sample digits.")
            
            # Display True label if we loaded a real digit
            if 'canvas_true_label' in st.session_state:
                st.markdown(f"""
                <div style="text-align: center; margin-top: 20px;">
                    <div style="font-size: 1rem; color: #666;">True Label:</div>
                    <div style="font-size: 3rem; font-weight: bold; color: #4e54c8;">{st.session_state.canvas_true_label}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display prediction results if available
        if 'prediction' in st.session_state:
            st.markdown("<h3>🔮 Prediction Results</h3>", unsafe_allow_html=True)
            
            result_cols = st.columns([1, 2])
            
            with result_cols[0]:
                st.markdown(f"""
                <div style="background:white; padding:30px; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1); text-align:center;">
                    <div style="font-size:1rem; color:#666; margin-bottom:10px;">Predicted Digit</div>
                    <div style="font-size:5rem; font-weight:700; color:#4e54c8;">{st.session_state.prediction['class']}</div>
                    <div style="font-size:1.2rem; color:#666; margin-top:10px;">Confidence: {st.session_state.prediction['confidence']*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_cols[1]:
                # Create bar chart of all probabilities
                fig = px.bar(
                    x=list(range(10)),
                    y=st.session_state.prediction['all_probs'],
                    labels={'x': 'Digit', 'y': 'Probability'},
                    color=st.session_state.prediction['all_probs'],
                    color_continuous_scale='viridis',
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    coloraxis_showscale=False,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <div style="text-align:center;">
        <h3>🧠 Deep Learning Playground</h3>
        <p style="color:#4e54c8; font-weight:600;">Developed with ❤️ by Shreyas Kasture for machine learning enthusiasts</p>
    </div>
</div>
""", unsafe_allow_html=True)
