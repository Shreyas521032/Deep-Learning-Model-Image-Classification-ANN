# heart_disease_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

# Set page config
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #F5F5F5;}
    h1 {color: #ff4b4b;}
    .stButton>button {background-color: #ff4b4b; color: white;}
    .stAlert {border-left-color: #ff4b4b !important;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("❤️ Heart Disease Risk Prediction using Deep Learning")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('heart disease risk.csv').rename(columns={'target': 'has_disease'})

df = load_data()

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    test_size = st.slider("Test Set Size (%)", 10, 30, 20)
    epochs = st.slider("Training Epochs", 50, 200, 100)
    batch_size = st.slider("Batch Size", 16, 64, 32)
    st.markdown("---")
    st.info("Configure model parameters using the sliders above.")

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

# Model Architecture
def create_model():
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    return model

model = create_model()

# Main Content
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(), height=200)
    st.write(f"Shape: {df.shape}")

with col2:
    st.subheader("📈 Class Distribution")
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(x='has_disease', data=df, palette='viridis')
    plt.xlabel("Heart Disease Presence")
    plt.ylabel("Count")
    st.pyplot(fig)

# Model Training
st.markdown("---")
st.subheader("🤖 Model Training")

if st.button("🚀 Start Training"):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    with st.spinner("Training model... This might take a while ⏳"):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=0
        )
    
    st.success("✅ Training Completed!")
    
    # Evaluation Metrics
    st.markdown("---")
    st.subheader("📊 Evaluation Metrics")
    
    loss, accuracy, auc_score = model.evaluate(X_test, y_test, verbose=0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Loss", f"{loss:.4f}")
    col2.metric("Accuracy", f"{accuracy:.4f}")
    col3.metric("AUC Score", f"{auc_score:.4f}")
    
    # Generate predictions
    y_pred_probs = model.predict(X_test).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Classification Report
    with st.expander("📝 Detailed Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(color='#ff4b4b', axis=0))
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 Visualization Dashboard")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
    
    # ROC Curve
    with col2:
        st.markdown("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        st.pyplot(fig)
    
    # Precision-Recall Curve
    with st.expander("📉 Precision-Recall Curve"):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
        fig = plt.figure()
        plt.plot(recall, precision, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        st.pyplot(fig)
    
    # Learning Curves
    with st.expander("📚 Learning Curves"):
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 16px;'>
        <em>Copyright © Shreyas Kasture, All rights reserved</em>
    </div>
""", unsafe_allow_html=True)
