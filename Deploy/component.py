import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
import time
from streamlit_lottie import st_lottie
import requests
import json
from PIL import Image

# Set page config
st.set_page_config(
    page_title="❤️ Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
        
# Load animations
heart_animation = load_gif("src/Animation - 1745652492296.gif")
loading_animation = load_gif("src/Animation - 1745652688329.gif")
success_animation = load_gif("src/Animation - 1745652761606.gif")

if loading_animation:
    st_lottie(loading_animation, 
             height=100,
             key="loading_data",
             speed=1,
             loop=True,
             quality="high")
else:
    st.warning("Couldn't load animation. Proceeding without it.")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        color: #e0e0e0;
    }
    h1, h2, h3 {
        background: linear-gradient(to right, #ff4b4b, #ff9e9e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff758c);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 12px 24px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(255, 117, 140, 0.5);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 14px rgba(255, 117, 140, 0.7);
    }
    .stAlert {
        border-left: 5px solid #ff4b4b !important;
        background-color: rgba(255, 75, 75, 0.1);
        border-radius: 10px;
    }
    .stProgress .st-bo {
        background-color: #ff4b4b;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: rgba(255, 75, 75, 0.1);
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #ff4b4b;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.07);
        border-radius: 0 0 10px 10px;
        padding: 20px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .css-1r6slb0 {
        border-radius: 10px !important;
        background-color: rgba(255, 255, 255, 0.07) !important;
    }
    .css-1r6slb0:hover {
        border: 2px solid #ff4b4b !important;
    }
    .css-145kmo2 {
        border-radius: 10px !important;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(to right, #ff4b4b, #ff9e9e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 1rem;
        color: #e0e0e0;
    }
    div[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.2);
        padding: 2rem 1rem;
        border-radius: 0 20px 20px 0;
    }
    .fancy-progress {
        margin: 20px 0;
        height: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        overflow: hidden;
    }
    .fancy-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff758c);
        animation: pulse 2s infinite;
        border-radius: 5px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .card {
        background-color: rgba(255, 255, 255, 0.07);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: rgba(0, 0, 0, 0.8);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .emoji-pulse {
        animation: emoji-pulse 1.5s infinite;
    }
    @keyframes emoji-pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar Animation
with st.sidebar:
    if heart_animation:
        st_lottie(heart_animation, height=200, key="heart_animation")
    else:
        st.warning("❤️ Animation unavailable")
    st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Model Parameters")
    test_size = st.slider("Test Set Size (%)", 10, 30, 20)
    
    epochs = st.slider("Training Epochs 🔄", 50, 200, 100)
    
    batch_sizes = [16, 32, 64, 128]
    batch_size = st.select_slider("Batch Size 📦", options=batch_sizes, value=32)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎨 Display Options")
    color_theme = st.selectbox("Chart Color Theme 🎭", 
                           ["plasma", "viridis", "inferno", "magma", "cividis"])
    show_detailed_metrics = st.checkbox("Show Advanced Metrics 📊", True)
    enable_animations = st.checkbox("Enable UI Animations ✨", True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Actions")
    if st.button("🔮 Random Hyperparameters"):
        st.session_state['test_size'] = np.random.randint(10, 31)
        st.session_state['epochs'] = np.random.randint(50, 201)
        st.session_state['batch_size'] = np.random.choice(batch_sizes)
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-top: 40px; text-align: center;'>
        <p style='opacity: 0.7;'>Made with ❤️ and AI</p>
    </div>
    """, unsafe_allow_html=True)

# Page Header with Animation
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='font-size: 3rem;'>❤️ Heart Disease Risk Wizard ❤️</h1>
    <p style='font-size: 1.2rem; color: #e0e0e0;'>Using Deep Learning to Predict Heart Disease Risk</p>
    <div style='background: rgba(255, 75, 75, 0.1); padding: 10px; border-radius: 10px; margin-top: 20px;'>
        <p style='margin: 0;'>👩‍⚕️ Helping healthcare professionals make better decisions 👨‍⚕️</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('Dataset/heart disease risk.csv').rename(columns={'target': 'has_disease'})

# Get data and show loading animation
with st.spinner("Loading dataset..."):
    if enable_animations:
        if loading_animation:
            st_lottie(loading_animation, height=100, key="loading_data")
            time.sleep(1.5)
        else:
            st.warning("⚠️ Couldn't load animation - proceeding without visual effects")
    df = load_data()
    if enable_animations:
        st.success("✅ Dataset loaded successfully!")
        time.sleep(0.5)

# Main Content
st.markdown("---")
tabs = st.tabs(["📊 Data Analysis", "🧠 Model Training", "🔍 Predictions", "ℹ️ About"])

# Data Analysis Tab
with tabs[0]:
    st.markdown("<h2>📊 Data Explorer</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10).style.background_gradient(cmap=color_theme, axis=0), height=300)
        st.markdown(f"<p>🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Class Distribution")
        fig = px.pie(df, 
                    names='has_disease', 
                    title='Heart Disease Distribution', 
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    hole=0.4)
        fig.update_layout(
            legend_title="Heart Disease",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        fig.update_traces(
            textinfo='percent+label', 
            textfont_size=14,
            marker=dict(line=dict(color='#1a1a2e', width=2))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Feature Correlation")
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=color_theme, linewidths=1)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Feature Distributions")
        feature = st.selectbox("Select Feature to Analyze 🔍", df.columns)
        
        fig = px.histogram(df, x=feature, color='has_disease', 
                        barmode='overlay', 
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        marginal='box')
        fig.update_layout(
            title=f"Distribution of {feature} by Heart Disease Status",
            xaxis_title=feature,
            yaxis_title="Count",
            legend_title="Heart Disease"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔄 Feature Relationships")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        feature_x = st.selectbox("X-axis Feature 📏", df.columns, index=0)
        feature_y = st.selectbox("Y-axis Feature 📐", df.columns, index=1)
    
    with col2:
        fig = px.scatter(df, x=feature_x, y=feature_y, 
                      color='has_disease', 
                      color_discrete_sequence=px.colors.sequential.Plasma_r,
                      opacity=0.7,
                      size_max=10,
                      trendline="ols",
                      hover_data=df.columns)
        
        fig.update_layout(
            title=f"Relationship between {feature_x} and {feature_y}",
            legend_title="Heart Disease",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Model Training Tab
with tabs[1]:
    st.markdown("<h2>🧠 Neural Network Training Center</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### 🧪 Experiment Setup
    
    Configure your model parameters and train your neural network to predict heart disease risk.
    """)
    
    # Data Preprocessing
    X = df.drop('has_disease', axis=1).values
    y = df['has_disease'].astype(int).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    
    # User-configurable model architecture
    st.markdown("### 🏗️ Model Architecture")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        architecture = st.radio("Select Model Complexity 🔧", 
                              ["Simple", "Moderate", "Complex"], 
                              horizontal=True)
    
    with col2:
        dropout_rate = st.slider("Dropout Rate 🎲", 0.0, 0.5, 0.3, 0.05)
    
    with col3:
        activation = st.selectbox("Activation Function ⚡", 
                               ["relu", "selu", "elu", "tanh"])
    
    # Define model architectures
    def create_model():
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        
        if architecture == "Simple":
            model.add(Dense(32, activation=activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(16, activation=activation))
            
        elif architecture == "Moderate":
            model.add(Dense(64, activation=activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation=activation))
            model.add(Dropout(dropout_rate/2))
            
        else:  # Complex
            model.add(Dense(128, activation=activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(64, activation=activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation=activation))
            model.add(Dropout(dropout_rate/2))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        return model
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Training
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🚀 Start Training")
    
    # Visualize model architecture
    model_viz_col1, model_viz_col2 = st.columns([2, 3])
    
    with model_viz_col1:
        if architecture == "Simple":
            layers = ["Input", "Dense(32)", "Dropout", "Dense(16)", "Output"]
        elif architecture == "Moderate":
            layers = ["Input", "Dense(64)", "Dropout", "Dense(32)", "Dropout", "Output"]
        else:  # Complex
            layers = ["Input", "Dense(128)", "Dropout", "Dense(64)", "Dropout", "Dense(32)", "Dropout", "Output"]
        
        st.markdown("#### 📝 Model Summary")
        for i, layer in enumerate(layers):
            if i == 0:
                st.markdown(f"🔹 **Layer {i+1}:** {layer} ({X_train.shape[1]} features)")
            elif i == len(layers) - 1:
                st.markdown(f"🔹 **Layer {i+1}:** {layer} (1 unit, sigmoid)")
            else:
                st.markdown(f"🔹 **Layer {i+1}:** {layer} ({activation})")
    
    with model_viz_col2:
        # Visual representation of NN architecture
        st.markdown("#### 🧠 Network Architecture")
        
        # Create nodes and connections for visualization
        layers_count = len(layers)
        layer_sizes = []
        
        if architecture == "Simple":
            layer_sizes = [X_train.shape[1], 32, 16, 1]
        elif architecture == "Moderate":
            layer_sizes = [X_train.shape[1], 64, 32, 1]
        else:  # Complex
            layer_sizes = [X_train.shape[1], 128, 64, 32, 1]
        
        fig = go.Figure()
        
        # Add nodes
        x_positions = np.linspace(0, 1, len(layer_sizes))
        max_neurons = max(layer_sizes)
        
        for i, (x_pos, n_neurons) in enumerate(zip(x_positions, layer_sizes)):
            # Calculate y positions for neurons in this layer
            y_positions = np.linspace(0, 1, n_neurons+2)[1:-1]
            
            # Add dots for neurons
            for y_pos in y_positions:
                fig.add_trace(go.Scatter(
                    x=[x_pos],
                    y=[y_pos],
                    mode='markers',
                    marker=dict(size=10, color='#ff4b4b'),
                    showlegend=False
                ))
            
            # Add layer label
            if i == 0:
                layer_name = "Input"
            elif i == len(layer_sizes) - 1:
                layer_name = "Output"
            else:
                layer_name = f"Hidden {i}"
                
            fig.add_annotation(
                x=x_pos,
                y=1.1,
                text=layer_name,
                showarrow=False,
                font=dict(size=12, color='white')
            )
            
            # Display neuron count
            fig.add_annotation(
                x=x_pos,
                y=-0.1,
                text=f"{n_neurons} units",
                showarrow=False,
                font=dict(size=10, color='white')
            )
            
            # Add connections to next layer
            if i < len(layer_sizes) - 1:
                next_x = x_positions[i+1]
                next_y_positions = np.linspace(0, 1, layer_sizes[i+1]+2)[1:-1]
                
                for y_pos in y_positions:
                    for next_y in next_y_positions:
                        fig.add_trace(go.Scatter(
                            x=[x_pos, next_x],
                            y=[y_pos, next_y],
                            mode='lines',
                            line=dict(color='rgba(255, 255, 255, 0.1)', width=1),
                            showlegend=False
                        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False, range=[-0.1, 1.1]),
            yaxis=dict(visible=False, range=[-0.2, 1.2]),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Training Button
    train_col1, train_col2, train_col3 = st.columns([1, 2, 1])
    
    with train_col2:
        if st.button("🔥 Train Neural Network 🔥"):
            with st.spinner("Preparing model for training..."):
                model = create_model()
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                # Show animated progress
                if enable_animations:
                    st_lottie(loading_animation, height=150, key="training_animation")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "auc": [], "val_auc": []}
                
                # Track training progress
                for epoch in range(epochs):
                    # Simulated training for smoother progress bar
                    history_temp = model.fit(
                        X_train, y_train,
                        epochs=1,
                        batch_size=batch_size,
                        validation_split=0.1,
                        class_weight=class_weight,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                    # Update progress bar
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    
                    # Update status text
                    current_loss = history_temp.history['loss'][0]
                    current_acc = history_temp.history['accuracy'][0]
                    status_text.markdown(f"""
                    <div style='text-align: center;'>
                        <p>Epoch {epoch+1}/{epochs} - Loss: {current_loss:.4f} - Accuracy: {current_acc:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Update history
                    for key in history:
                        if key in history_temp.history:
                            history[key].append(history_temp.history[key][0])
                    
                    # Check for early stopping
                    if len(history['val_loss']) > 5:
                        if all(history['val_loss'][-5-1] <= history['val_loss'][-i] for i in range(1, 6)):
                            status_text.markdown(f"""
                            <div style='text-align: center;'>
                                <p>Early stopping triggered at epoch {epoch+1}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            break
                    
                    time.sleep(0.1)  # Small delay for visual effect
                
                # Show completion animation
                if enable_animations:
                    status_text.empty()
                    st_lottie(success_animation, height=150, key="success_animation")
                
                st.success("✅ Training completed successfully!")
                
                # Store model and history in session state
                st.session_state['model'] = model
                st.session_state['history'] = history
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['trained'] = True
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Results section (only shown after training)
    if 'trained' in st.session_state and st.session_state['trained']:
        st.markdown("---")
        st.markdown("<h2>🎯 Training Results</h2>", unsafe_allow_html=True)
        
        model = st.session_state['model']
        history = st.session_state['history']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Evaluation Metrics
        loss, accuracy, auc_score = model.evaluate(X_test, y_test, verbose=0)
        
        # Fancy metric display
        st.markdown("""
        <div style='display: flex; justify-content: space-around; margin-bottom: 30px;'>
            <div class='metric-card'>
                <div class='emoji-pulse' style='font-size: 2rem;'>📊</div>
                <div class='metric-value'>{:.1f}%</div>
                <div class='metric-label'>Accuracy</div>
            </div>
            <div class='metric-card'>
                <div class='emoji-pulse' style='font-size: 2rem;'>🎯</div>
                <div class='metric-value'>{:.3f}</div>
                <div class='metric-label'>AUC Score</div>
            </div>
            <div class='metric-card'>
                <div class='emoji-pulse' style='font-size: 2rem;'>📉</div>
                <div class='metric-value'>{:.3f}</div>
                <div class='metric-label'>Loss</div>
            </div>
        </div>
        """.format(accuracy*100, auc_score, loss), unsafe_allow_html=True)
        
        # Generate predictions
        y_pred_probs = model.predict(X_test).ravel()
        y_pred = (y_pred_probs >= 0.5).astype(int)
        
        # Results visualization
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Performance Visualizations")
        
        # Tabs for different visualizations
        viz_tabs = st.tabs(["Learning Curves", "Confusion Matrix", "ROC Curve", "Prediction Distribution"])
        
        with viz_tabs[0]:
            # Learning Curves
            fig = go.Figure()
            
            # Loss curve
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history['loss'])+1)),
                y=history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='#ff4b4b', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history['val_loss'])+1)),
                y=history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='#ff9e9e', width=2, dash='dash')
            ))
            
            # Accuracy curve
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history['accuracy'])+1)),
                y=history['accuracy'],
                mode='lines',
                name='Training Accuracy',
                line=dict(color='#4b7bff', width=2),
                yaxis='y2'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history['val_accuracy'])+1)),
                y=history['val_accuracy'],
                mode='lines',
                name='Validation Accuracy',
                line=dict(color='#9ebaff', width=2, dash='dash'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Learning Curves',
                xaxis=dict(title='Epoch'),
                yaxis=dict(
                    title='Loss',
                    titlefont=dict(color='#ff4b4b'),
                    tickfont=dict(color='#ff4b4b')
                ),
                yaxis2=dict(
                    title='Accuracy',
                    titlefont=dict(color='#4b7bff'),
                    tickfont=dict(color='#4b7bff'),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                legend=dict(
                    bgcolor='rgba(255, 255, 255, 0.1)',
                    bordercolor='rgba(255, 255, 255, 0.2)',
                    borderwidth=1
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Disease', 'Disease'],
                y=['No Disease', 'Disease'],
                color_continuous_scale=color_theme
            )
            
            fig.update_layout(
                title='Confusion Matrix',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                height=500
            )
            
            # Add annotations with percentages
            total = cm.sum()
            for i in range(2):
                for j in range(2):
                    percentage = 100 * cm[i, j] / total
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=f"{cm[i, j]} ({percentage:.1f}%)",
                        showarrow=False,
                        font=dict(color="white", size=14)
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics if enabled
            if show_detailed_metrics:
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                st.markdown("#### Detailed Classification Metrics")
                st.dataframe(report_df.style.background_gradient(cmap=color_theme, axis=0), height=200)
        
        with viz_tabs[2]:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='#ff4b4b', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 75, 75, 0.1)'
            ))
            
            # Diagonal line (random classifier)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC)',
                xaxis=dict(title='False Positive Rate', constrain='domain'),
                yaxis=dict(title='True Positive Rate', scaleanchor="x", scaleratio=1),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                height=500,
                shapes=[
                    dict(
                        type='rect',
                        xref='paper',
                        yref='paper',
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=1,
                        line=dict(width=1, color='rgba(255, 255, 255, 0.2)')
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Precision-Recall curve if advanced metrics enabled
            if show_detailed_metrics:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
                pr_auc = auc(recall, precision)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'PR Curve (AUC = {pr_auc:.3f})',
                    line=dict(color='#4b7bff', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(75, 123, 255, 0.1)'
                ))
                
                # Add baseline (prevalence)
                baseline = sum(y_test) / len(y_test)
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[baseline, baseline],
                    mode='lines',
                    name=f'Baseline ({baseline:.3f})',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis=dict(title='Recall'),
                    yaxis=dict(title='Precision'),
                    legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.1)'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0'),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[3]:
            # Prediction Distribution
            fig = go.Figure()
            
            # Add distribution for class 0
            fig.add_trace(go.Histogram(
                x=y_pred_probs[y_test == 0],
                name='No Disease (Actual)',
                opacity=0.7,
                marker_color='#4b7bff',
                bingroup='shared'
            ))
            
            # Add distribution for class 1
            fig.add_trace(go.Histogram(
                x=y_pred_probs[y_test == 1],
                name='Disease (Actual)',
                opacity=0.7,
                marker_color='#ff4b4b',
                bingroup='shared'
            ))
            
            # Add threshold line
            fig.add_shape(
                type='line',
                x0=0.5, y0=0,
                x1=0.5, y1=1,
                yref='paper',
                line=dict(color='white', width=2, dash='dash')
            )
            
            fig.add_annotation(
                x=0.5, y=1.05,
                yref='paper',
                text='Classification Threshold',
                showarrow=False,
                font=dict(color='white')
            )
            
            fig.update_layout(
                title='Prediction Probability Distribution',
                xaxis=dict(title='Predicted Probability', range=[0, 1]),
                yaxis=dict(title='Count'),
                barmode='overlay',
                bargap=0.1,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                height=500,
                legend=dict(
                    bgcolor='rgba(255, 255, 255, 0.1)',
                    bordercolor='rgba(255, 255, 255, 0.2)',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Predictions Tab
with tabs[2]:
    st.markdown("<h2>🔍 Heart Disease Risk Predictor</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### 🏥 Patient Risk Assessment
    
    Enter patient information to predict heart disease risk. This tool helps healthcare professionals make informed decisions.
    """)
    
    # Check if model exists
    model_exists = 'model' in st.session_state
    
    if not model_exists:
        st.warning("⚠️ Please train the model in the 'Model Training' tab first!")
    else:
        # Input form for prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 100, 50)
            sex = st.radio("Sex", ["Female", "Male"])
            sex = 0 if sex == "Female" else 1
            cp = st.selectbox("Chest Pain Type", 
                          ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                          index=0)
            cp_values = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            cp = cp_values[cp]
            
        with col2:
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.slider("Cholesterol (mg/dl)", 100, 600, 250)
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            fbs = 0 if fbs == "No" else 1
            
        with col3:
            restecg = st.selectbox("Resting ECG", 
                               ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                               index=0)
            restecg_values = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            restecg = restecg_values[restecg]
            thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
            exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
            exang = 0 if exang == "No" else 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                             ["Upsloping", "Flat", "Downsloping"],
                             index=0)
            slope_values = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope = slope_values[slope]
            
        with col2:
            ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
            thal = st.selectbox("Thalassemia", 
                            ["Normal", "Fixed Defect", "Reversible Defect"],
                            index=0)
            thal_values = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
            thal = thal_values[thal]
        
        # Make prediction
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        
        with predict_col2:
            if st.button("🔮 Predict Risk"):
                with st.spinner("Analyzing patient data..."):
                    # Create input array
                    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                          thalach, exang, oldpeak, slope, ca, thal]])
                    
                    # Scale input
                    scaled_input = scaler.transform(input_data)
                    
                    # Show animated prediction process
                    if enable_animations:
                        st_lottie(loading_animation, height=100, key="predict_animation")
                        time.sleep(1)  # For effect
                    
                    # Get prediction
                    prediction = model.predict(scaled_input)[0][0]
                    risk_percentage = prediction * 100
                    
                    # Display result with animation
                    if enable_animations:
                        progress_width = min(risk_percentage, 100)
                        
                        st.markdown(f"""
                        <div style='text-align: center; margin: 20px 0;'>
                            <h3>Heart Disease Risk: {risk_percentage:.1f}%</h3>
                            <div class='fancy-progress'>
                                <div class='fancy-progress-bar' style='width: {progress_width}%;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk category
                    if risk_percentage < 25:
                        risk_category = "Low Risk"
                        emoji = "✅"
                        color = "green"
                    elif risk_percentage < 50:
                        risk_category = "Moderate Risk"
                        emoji = "⚠️"
                        color = "orange"
                    elif risk_percentage < 75:
                        risk_category = "High Risk"
                        emoji = "🚨"
                        color = "red"
                    else:
                        risk_category = "Very High Risk"
                        emoji = "⛔"
                        color = "darkred"
                    
                    st.markdown(f"""
                    <div style='text-align: center; margin: 20px 0;'>
                        <h2 style='color: {color};'>{emoji} {risk_category} {emoji}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show risk factors if high risk
                    if risk_percentage >= 50:
                        st.markdown("### Key Risk Factors")
                        
                        risk_factors = []
                        
                        if age > 60:
                            risk_factors.append("Advanced age")
                        if sex == 1:
                            risk_factors.append("Male gender")
                        if cp == 3:
                            risk_factors.append("Asymptomatic chest pain")
                        if trestbps > 140:
                            risk_factors.append("High blood pressure")
                        if chol > 240:
                            risk_factors.append("High cholesterol")
                        if fbs == 1:
                            risk_factors.append("High fasting blood sugar")
                        if thalach < 120:
                            risk_factors.append("Low maximum heart rate")
                        if exang == 1:
                            risk_factors.append("Exercise-induced angina")
                        if oldpeak > 2:
                            risk_factors.append("Significant ST depression")
                        if ca > 0:
                            risk_factors.append(f"{ca} major vessels colored by fluoroscopy")
                        if thal == 3:
                            risk_factors.append("Reversible defect in thalassemia")
                        
                        for i, factor in enumerate(risk_factors):
                            st.markdown(f"- {factor}")
                        
                        st.markdown("""
                        <div style='background-color: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px;'>
                            <p><strong>⚕️ Medical Disclaimer:</strong> This prediction is meant to assist healthcare professionals and should not replace proper medical diagnosis. Please consult with a qualified healthcare provider for proper evaluation.</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# About Tab
with tabs[3]:
    st.markdown("<h2>ℹ️ About This Application</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### 🔬 Technical Details
    
    This application uses a neural network to predict the likelihood of heart disease based on patient data. The model is trained on a dataset of patient records with known heart disease status.
    
    #### Dataset Features:
    
    - **age**: Age in years
    - **sex**: Gender (0 = female, 1 = male)
    - **cp**: Chest pain type (0-3)
    - **trestbps**: Resting blood pressure in mm Hg
    - **chol**: Serum cholesterol in mg/dl
    - **fbs**: Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
    - **restecg**: Resting electrocardiographic results (0-2)
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (0 = no, 1 = yes)
    - **oldpeak**: ST depression induced by exercise relative to rest
    - **slope**: Slope of the peak exercise ST segment (0-2)
    - **ca**: Number of major vessels colored by fluoroscopy (0-4)
    - **thal**: Thalassemia (1-3)
    
    #### Model Architecture:
    
    The neural network architecture can be customized in the Model Training tab. The default architecture includes:
    
    - Input layer with same dimension as features
    - Multiple hidden layers with configurable size and activation functions
    - Dropout layers for regularization
    - Output layer with sigmoid activation for binary classification
    
    #### Performance Metrics:
    
    The model is evaluated using:
    
    - Accuracy: Percentage of correct predictions
    - AUC: Area Under the ROC Curve, measures discrimination capability
    - Loss: Binary cross-entropy loss function
    
    Additionally, detailed metrics such as precision, recall, and F1-score are available.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### 📚 Educational Purpose
    
    This tool is designed for educational purposes and to demonstrate:
    
    1. How machine learning can be applied to healthcare problems
    2. The process of model development and evaluation
    3. Interactive data visualization and exploration
    4. The importance of feature selection and preprocessing in medical data
    
    ### ⚠️ Medical Disclaimer
    
    This application is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    
    The predictions made by this model should be used only as a supplementary tool by healthcare professionals. Clinical judgment and established medical protocols should always take precedence.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
