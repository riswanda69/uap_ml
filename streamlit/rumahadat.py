import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import gdown
import os

# Set page config
st.set_page_config(
    page_title="Rumah Adat Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Class mapping for both models
CLASS_NAMES = ['gadang', 'honai', 'joglo', 'panjang', 'tongkonan']

@st.cache_resource
def load_models():
    """Load CNN and ResNet models directly from Google Drive."""


    # ID file Google Drive
    cnn_model_id = "1eFanfchOz7ESHwfyX8B8GhpF4SAYvfGX"
    resnet_model_id = "1ZNS5DSn5vQAKh1HjQo5IdjPHibAcnxDt"

    # URL download Google Drive
    cnn_model_url = f"https://drive.google.com/uc?id={cnn_model_id}"
    resnet_model_url = f"https://drive.google.com/uc?id={resnet_model_id}"

    # Unduh model ke memori
    cnn_model_bytes = gdown.download(cnn_model_url, None, quiet=False)
    resnet_model_bytes = gdown.download(resnet_model_url, None, quiet=False)

    # Muat model dari memori
    cnn_model = tf.keras.models.load_model(cnn_model_bytes)
    resnet_model = tf.keras.models.load_model(resnet_model_bytes)

    return cnn_model, resnet_model

def preprocess_image_cnn(image):
    """Preprocess image for CNN model"""
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_resnet(image):
    """Preprocess image for ResNet model"""
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_batch(images, model, preprocess_func):
    """Predict a batch of images"""
    predictions = []
    for img in images:
        processed_img = preprocess_func(img)
        pred = model.predict(processed_img, verbose=0)
        predictions.append(pred[0])
    return np.array(predictions)

def plot_prediction_confidence(predictions, class_names):
    """Create confidence plot using plotly"""
    fig = go.Figure()
    
    top_preds_idx = np.argsort(predictions)[-5:][::-1]
    
    fig.add_trace(go.Bar(
        x=[class_names[idx] for idx in top_preds_idx],
        y=[predictions[idx] * 100 for idx in top_preds_idx],
        marker_color='rgba(255, 75, 75, 0.8)',
    ))
    
    fig.update_layout(
        title="Top 5 Predictions Confidence",
        xaxis_title="Class",
        yaxis_title="Confidence (%)",
        template="plotly_white",
        height=400,
    )
    
    return fig

def main():
    st.title("🎨 Rumah Adat Pattern Classification")
    st.write("Upload Rumah Adat images to classify their patterns using CNN and ResNet models")
    
    # Load models
    try:
        cnn_model, resnet_model = load_models()
        st.sidebar.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose Rumah Adat images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create columns for model selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("### CNN Model")
            use_cnn = st.checkbox("Use CNN Model", value=True)
            
        with col2:
            st.info("### ResNet Model")
            use_resnet = st.checkbox("Use ResNet Model", value=True)
        
        if st.button("Analyze Images"):
            if not (use_cnn or use_resnet):
                st.warning("Please select at least one model!")
                return
            
            with st.spinner("Processing images..."):
                # Process each image
                for idx, file in enumerate(uploaded_files):
                    st.markdown("---")
                    st.subheader(f"Image {idx + 1}")
                    
                    # Display image
                    img = Image.open(file)
                    st.image(img, width=300)
                    
                    # Create columns for predictions
                    if use_cnn and use_resnet:
                        col1, col2 = st.columns(2)
                    else:
                        col1 = st
                    
                    # CNN Predictions
                    if use_cnn:
                        with col1:
                            st.markdown("#### CNN Model Prediction")
                            processed_img = preprocess_image_cnn(img)
                            cnn_pred = cnn_model.predict(processed_img, verbose=0)[0]
                            cnn_class_idx = np.argmax(cnn_pred)
                            
                            # Display prediction
                            st.markdown(
                                f"""
                                <div class="prediction-box">
                                    <h4>Predicted: {CLASS_NAMES[cnn_class_idx]}</h4>
                                    <p>Confidence: {cnn_pred[cnn_class_idx]*100:.2f}%</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Plot confidence with unique key
                            st.plotly_chart(
                                plot_prediction_confidence(cnn_pred, CLASS_NAMES),
                                use_container_width=True,
                                key=f"cnn_plot_{idx}"  # Added unique key
                            )
                    
                    # ResNet Predictions
                    if use_resnet:
                        with col2 if use_cnn else col1:
                            st.markdown("#### ResNet Model Prediction")
                            processed_img = preprocess_image_resnet(img)
                            resnet_pred = resnet_model.predict(processed_img, verbose=0)[0]
                            resnet_class_idx = np.argmax(resnet_pred)
                            
                            # Display prediction
                            st.markdown(
                                f"""
                                <div class="prediction-box">
                                    <h4>Predicted: {CLASS_NAMES[resnet_class_idx]}</h4>
                                    <p>Confidence: {resnet_pred[resnet_class_idx]*100:.2f}%</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Plot confidence with unique key
                            st.plotly_chart(
                                plot_prediction_confidence(resnet_pred, CLASS_NAMES),
                                use_container_width=True,
                                key=f"resnet_plot_{idx}"  # Added unique key
                            )
                    
                    # Model agreement check
                    if use_cnn and use_resnet:
                        if cnn_class_idx == resnet_class_idx:
                            st.success("✅ Both models agree on the prediction!")
                        else:
                            st.warning("⚠ Models have different predictions. Consider reviewing the image.")
                
            st.success("✨ All images processed successfully!")
    
    # Add sidebar information
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("""
    #### CNN Model
    - Custom architecture
    - Trained from scratch
    - Good for basic pattern recognition
    
    #### ResNet Model
    - Based on ResNet50
    - Transfer learning
    - Better for complex patterns
    
    ### Tips for Better Results
    1. Use clear, well-lit images
    2. Ensure pattern is visible
    3. Avoid blurry or distorted images
    4. Compare both models' predictions
    """)

if __name__ == "__main__":
    main()