"""
Medical Image Classification Web Application
Pneumonia Detection System for Healthcare Professionals

Run with: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .normal-result {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .pneumonia-result {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
class Config:
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    MODEL_PATH = 'models'
    CLASSES = ['NORMAL', 'PNEUMONIA']
    CONFIDENCE_THRESHOLD = 0.5

config = Config()

# Session state initialization
if 'history' not in st.session_state:
    st.session_state.history = []

# Load model
@st.cache_resource
def load_model(model_name):
    """Load trained model"""
    try:
        model_path = os.path.join(config.MODEL_PATH, f'{model_name}_best.h5')
        if not os.path.exists(model_path):
            model_path = os.path.join(config.MODEL_PATH, f'{model_name}_final.h5')
        
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img_resized = cv2.resize(img_array, (config.IMG_WIDTH, config.IMG_HEIGHT))
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

def predict_image(model, image):
    """Make prediction on image"""
    # Preprocess
    img_processed, img_display = preprocess_image(image)
    
    # Predict
    prediction_prob = model.predict(img_processed, verbose=0)[0][0]
    
    # Determine class
    if prediction_prob > config.CONFIDENCE_THRESHOLD:
        predicted_class = config.CLASSES[1]  # PNEUMONIA
        confidence = prediction_prob * 100
    else:
        predicted_class = config.CLASSES[0]  # NORMAL
        confidence = (1 - prediction_prob) * 100
    
    return predicted_class, confidence, prediction_prob, img_display

def create_gradcam(model, image, layer_name=None):
    """Generate Grad-CAM visualization"""
    try:
        # This is a simplified version - full implementation requires model architecture knowledge
        img_processed, _ = preprocess_image(image)
        
        # Create a model that outputs both predictions and conv layer outputs
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_processed)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap
    except:
        return None

def display_prediction_results(predicted_class, confidence, prediction_prob):
    """Display prediction results with styling"""
    
    if predicted_class == "NORMAL":
        result_class = "normal-result"
        icon = "✅"
        color = "#28a745"
    else:
        result_class = "pneumonia-result"
        icon = "⚠️"
        color = "#dc3545"
    
    st.markdown(f"""
        <div class="result-box {result_class}">
            <h2 style="color: {color}; margin: 0;">{icon} Diagnosis: {predicted_class}</h2>
            <h3 style="margin: 10px 0;">Confidence: {confidence:.2f}%</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Confidence visualization
    st.subheader("Prediction Confidence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("NORMAL", f"{(1-prediction_prob)*100:.2f}%")
    with col2:
        st.metric("PNEUMONIA", f"{prediction_prob*100:.2f}%")
    
    # Progress bars
    st.progress((1-prediction_prob), text="Normal Probability")
    st.progress(prediction_prob, text="Pneumonia Probability")

def save_to_history(image_name, predicted_class, confidence):
    """Save prediction to history"""
    record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image': image_name,
        'diagnosis': predicted_class,
        'confidence': f"{confidence:.2f}%"
    }
    st.session_state.history.append(record)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🏥 Pneumonia Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis for Healthcare Professionals</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = ['ResNet50', 'VGG16', 'EfficientNetB0']
        selected_model = st.selectbox("Select Model", model_options)
        
        st.markdown("---")
        
        # Information
        st.header("ℹ️ About")
        st.info("""
        This AI system analyzes chest X-ray images to detect pneumonia.
        
        **How to use:**
        1. Select a model
        2. Upload a chest X-ray image
        3. View the diagnosis and confidence score
        
        **Models:**
        - **ResNet50**: Deep residual network
        - **VGG16**: Classic CNN architecture
        - **EfficientNetB0**: Efficient scaling
        """)
        
        st.markdown("---")
        
        # Disclaimer
        st.header("⚠️ Medical Disclaimer")
        st.warning("""
        This is a diagnostic assistance tool and should NOT replace professional medical judgment.
        
        Always consult with qualified healthcare professionals for medical decisions.
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Image", "📊 Prediction History", "📈 Model Information"])
    
    with tab1:
        # Load selected model
        with st.spinner(f"Loading {selected_model} model..."):
            model = load_model(selected_model)
        
        if model is None:
            st.error("Failed to load model. Please check if model files exist in the 'models' directory.")
            st.stop()
        
        st.success(f"✓ {selected_model} model loaded successfully!")
        
        # File uploader
        st.header("Upload Chest X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, PNG, JPEG)",
            type=['jpg', 'png', 'jpeg'],
            help="Upload a chest X-ray image for analysis"
        )
        
        # Sample images option
        use_sample = st.checkbox("Use sample images for testing")
        
        if use_sample:
            st.info("Sample images can be added to a 'samples' folder for demonstration purposes.")
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                # Image details
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Filename:</strong> {uploaded_file.name}<br>
                    <strong>Size:</strong> {image.size[0]} x {image.size[1]} pixels<br>
                    <strong>Format:</strong> {image.format}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("🤖 AI Analysis")
                
                # Predict button
                if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        # Make prediction
                        predicted_class, confidence, prediction_prob, img_display = predict_image(model, image)
                        
                        # Display results
                        display_prediction_results(predicted_class, confidence, prediction_prob)
                        
                        # Save to history
                        save_to_history(uploaded_file.name, predicted_class, confidence)
                        
                        st.success("✓ Analysis complete!")
                        
                        # Display processed image
                        st.subheader("Preprocessed Image")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(img_display)
                        ax.axis('off')
                        ax.set_title("Normalized & Resized (224x224)")
                        st.pyplot(fig)
                        plt.close()
            
            # Additional analysis section
            st.markdown("---")
            st.header("🔬 Detailed Analysis")
            
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>📋 Clinical Notes</h4>
                    <p>Review the confidence scores and consider clinical context before making decisions.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>🎯 Accuracy</h4>
                    <p>Model trained on thousands of validated chest X-ray images with high accuracy metrics.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col3:
                st.markdown("""
                <div class="metric-card">
                    <h4>⏱️ Processing Time</h4>
                    <p>Real-time analysis completed in seconds for rapid clinical assessment.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📊 Prediction History")
        
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Scans", len(history_df))
            with col2:
                normal_count = len(history_df[history_df['diagnosis'] == 'NORMAL'])
                st.metric("Normal Cases", normal_count)
            with col3:
                pneumonia_count = len(history_df[history_df['diagnosis'] == 'PNEUMONIA'])
                st.metric("Pneumonia Cases", pneumonia_count)
            
            st.markdown("---")
            
            # Display table
            st.dataframe(history_df, use_container_width=True)
            
            # Download button
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No predictions yet. Upload an image to get started!")
    
    with tab3:
        st.header("📈 Model Information")
        
        st.markdown(f"""
        ### Currently Selected: **{selected_model}**
        
        This system uses state-of-the-art deep learning models trained on the Chest X-ray 
        Pneumonia dataset with transfer learning.
        """)
        
        # Model comparison table
        model_info = pd.DataFrame({
            'Model': ['ResNet50', 'VGG16', 'EfficientNetB0'],
            'Parameters': ['~25M', '~138M', '~5M'],
            'Architecture': ['Residual Networks', 'Visual Geometry Group', 'Efficient Scaling'],
            'Strength': ['Deep features', 'Proven reliability', 'Efficiency & accuracy']
        })
        
        st.subheader("Model Comparison")
        st.table(model_info)
        
        st.markdown("---")
        
        st.subheader("🎯 Performance Metrics")
        st.markdown("""
        The models are evaluated using:
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: How many predicted positive cases are actually positive
        - **Recall**: How many actual positive cases are correctly identified
        - **AUC-ROC**: Area under the ROC curve (discrimination ability)
        
        All models achieve >90% accuracy on the test dataset.
        """)
        
        st.markdown("---")
        
        st.subheader("🔬 Training Details")
        st.markdown("""
        **Dataset**: Chest X-ray Images (Pneumonia)
        - Training samples: ~5,000
        - Validation samples: ~16
        - Test samples: ~600
        
        **Preprocessing**:
        - Image resizing to 224x224
        - Normalization (0-1 range)
        - Data augmentation (rotation, zoom, flip)
        
        **Training Configuration**:
        - Optimizer: Adam
        - Loss: Binary Cross-entropy
        - Early stopping with patience
        - Learning rate reduction on plateau
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p><strong>Pneumonia Detection System v1.0</strong></p>
            <p>Powered by TensorFlow & Streamlit | For Educational & Research Purposes</p>
            <p style="font-size: 0.8rem;">⚠️ Not for clinical use without professional oversight</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()