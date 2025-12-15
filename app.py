import streamlit as st
import torch
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time

# Import inference functions
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import inference function
try:
    from src.inference import predict
except ImportError:
    st.error("Could not import inference module. Make sure src/inference.py exists.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ForensicaAI - Deepfake Detection System",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fake-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .confidence-meter {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎭 ForensicaAI - Deepfake Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model info
with st.sidebar:
    st.header("📊 Model Information")
    st.info("""
    **Model Architecture:**
    - CNN: MobileNetV2 (Feature Extractor)
    - RNN: LSTM Classifier
    - Input: 20 frames per video
    - Resolution: 128x128
    """)
    
    # Check if model exists
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        st.success("✅ Model loaded successfully")
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            if "best_val_acc" in checkpoint:
                st.metric("Best Validation Accuracy", f"{checkpoint['best_val_acc']:.2%}")
        except:
            pass
    else:
        st.error("❌ Model not found. Please train the model first.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze for deepfake detection"
    )
    
    if uploaded_file is not None:
        # Display video info
        st.subheader("📹 Video Information")
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }
        st.json(file_details)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_video_path = tmp_file.name
        
        # Display video preview
        st.subheader("🎬 Video Preview")
        st.video(uploaded_file)

with col2:
    st.header("🔍 Analysis Results")
    
    if uploaded_file is not None:
        if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Processing video... This may take a moment."):
                try:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Load model and extract frames
                    status_text.text("Step 1/2: Extracting frames from video...")
                    progress_bar.progress(30)
                    
                    # Step 2: Run prediction
                    status_text.text("Step 2/2: Analyzing with deepfake detection model...")
                    progress_bar.progress(60)
                    
                    # Run prediction (this handles frame extraction internally)
                    prediction, confidence = predict(tmp_video_path)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store results in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['confidence'] = confidence
                    st.session_state['video_path'] = tmp_video_path
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
        
        # Display results if available
        if 'prediction' in st.session_state and 'confidence' in st.session_state:
            prediction = st.session_state['prediction']
            confidence = st.session_state['confidence']
            
            # Prediction box
            box_class = "real-box" if prediction == "REAL" else "fake-box"
            icon = "✅" if prediction == "REAL" else "⚠️"
            color = "#28a745" if prediction == "REAL" else "#dc3545"
            
            st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2 style="text-align: center; color: {color};">
                        {icon} Prediction: <strong>{prediction}</strong>
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence score
            st.subheader("📊 Confidence Score")
            confidence_percent = confidence * 100
            
            # Confidence bar
            st.progress(confidence, text=f"{confidence_percent:.2f}%")
            
            # Confidence metric
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence", f"{confidence_percent:.2f}%")
            with col_b:
                uncertainty = (1 - confidence) * 100
                st.metric("Uncertainty", f"{uncertainty:.2f}%")
            
            # Detailed Report
            st.markdown("---")
            st.subheader("📋 Detailed Report")
            
            report_data = {
                "Video File": uploaded_file.name,
                "Prediction": prediction,
                "Confidence": f"{confidence_percent:.2f}%",
                "Model": "MobileNetV2 + LSTM",
                "Frames Analyzed": len(st.session_state.get('frames', [])),
                "Analysis Date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.json(report_data)
            
            # Interpretation
            st.subheader("💡 Interpretation")
            if prediction == "REAL":
                st.success(f"""
                The model predicts this video is **REAL** with {confidence_percent:.1f}% confidence.
                This suggests the video shows authentic, unmanipulated content.
                """)
            else:
                st.warning(f"""
                The model predicts this video is **FAKE** (deepfake) with {confidence_percent:.1f}% confidence.
                This suggests the video may contain AI-generated or manipulated content.
                """)
            
            # Video info
            st.subheader("📹 Video Analyzed")
            st.info(f"Video file: {uploaded_file.name} | 20 frames extracted and analyzed")
            
            # Download report button
            st.markdown("---")
            report_text = f"""
FORENSICAAI - DEEPFAKE DETECTION REPORT
========================================

Video File: {uploaded_file.name}
Analysis Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION: {prediction}
Confidence: {confidence_percent:.2f}%

Model Details:
- Architecture: MobileNetV2 + LSTM
- Frames Analyzed: 20
- Input Resolution: 128x128

Interpretation:
{'The video appears to be authentic and unmanipulated.' if prediction == 'REAL' else 'The video shows signs of AI-generated or manipulated content (deepfake).'}

---
Generated by ForensicaAI - Deepfake Detection System
            """
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"deepfake_report_{uploaded_file.name}_{int(time.time())}.txt",
                mime="text/plain"
            )
            
            # Cleanup temp file
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>ForensicaAI - Deepfake Detection System | Powered by CNN + LSTM Architecture</p>
    <p><small>For research and educational purposes</small></p>
</div>
""", unsafe_allow_html=True)

