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
                    height=500
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
