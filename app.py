import streamlit as st
import torch
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time
from torchvision import transforms

# Import inference functions
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import model components
try:
    from src.models.cnn_feature_extractor import build_cnn
    from src.models.rnn_classifier import RNNClassifier
except ImportError as e:
    st.error(f"Could not import model modules: {str(e)}")
    st.stop()

# Settings - Use absolute path resolution for model
MODEL_PATH = str(Path(__file__).parent / "models" / "best_model.pth")
FRAME_SIZE = 128
NUM_FRAMES = 20
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

@st.cache_resource
def load_model():
    """Load and cache the trained model with all weights and states."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Please ensure the model file exists. Current working directory: {os.getcwd()}"
            )
        
        # Load checkpoint
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {MODEL_PATH}: {str(e)}")
        
        # Get feature dimension from checkpoint
        if "feature_dim" not in checkpoint:
            raise KeyError(
                "Checkpoint missing 'feature_dim' key. "
                "Please ensure you're using a model trained with the current codebase."
            )
        
        feature_dim = checkpoint["feature_dim"]
        
        # Build CNN
        try:
            cnn, _ = build_cnn()
        except Exception as e:
            raise RuntimeError(f"Failed to build CNN: {str(e)}")
        
        # Build RNN with same parameters as training
        try:
            rnn = RNNClassifier(
                feature_dim=feature_dim,
                hidden_size=128,
                num_layers=1,
                bidirectional=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build RNN: {str(e)}")
        
        # Load state dicts
        if "cnn_state" not in checkpoint:
            raise KeyError("Checkpoint missing 'cnn_state' key")
        if "rnn_state" not in checkpoint:
            raise KeyError("Checkpoint missing 'rnn_state' key")
        
        try:
            cnn.load_state_dict(checkpoint["cnn_state"])
            rnn.load_state_dict(checkpoint["rnn_state"])
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
        
        # Move to device
        cnn.to(device)
        rnn.to(device)
        
        # Set to eval mode
        cnn.eval()
        rnn.eval()
        
        return cnn, rnn, device, checkpoint
    
    except Exception as e:
        st.error(f"Error in load_model(): {str(e)}")
        raise

def extract_frames(video_path):
    """Extract frames from video for inference."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total < NUM_FRAMES:
        raise ValueError(f"Video too short! Need at least {NUM_FRAMES} frames, got {total}.")
    
    interval = max(1, total // NUM_FRAMES)
    frames = []
    
    for i in range(NUM_FRAMES):
        frame_pos = min(i * interval, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < NUM_FRAMES:
        raise ValueError(f"Could only extract {len(frames)} frames, need {NUM_FRAMES}.")
    
    return frames

def predict_video(video_path, cnn, rnn, device):
    """Run prediction on a video using the loaded model."""
    # Extract frames
    frames = extract_frames(video_path)
    
    # Preprocess frames
    tensors = []
    for f in frames:
        tensors.append(transform(f))
    
    seq_tensor = torch.stack(tensors, dim=0)  # [T, C, H, W]
    seq_tensor = seq_tensor.unsqueeze(0)      # [1, T, C, H, W]
    
    # Predict
    with torch.no_grad():
        B, T, C, H, W = seq_tensor.shape
        seq_tensor = seq_tensor.to(device)
        
        reshaped = seq_tensor.view(B*T, C, H, W)
        features = cnn(reshaped)
        features = features.view(B, T, -1)
        
        logits = rnn(features)
        probs = torch.softmax(logits, dim=1)
    
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    
    label_map = {0: "REAL", 1: "FAKE"}
    
    return label_map[pred_class], confidence

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
    
    # Load model (cached)
    try:
        cnn, rnn, device, checkpoint = load_model()
        st.success("✅ Model loaded successfully")
        if "best_val_acc" in checkpoint:
            st.metric("Best Validation Accuracy", f"{checkpoint['best_val_acc']:.2%}")
        else:
            st.info("Model weights loaded (no accuracy metric in checkpoint)")
        st.metric("Device", str(device))
        if "feature_dim" in checkpoint:
            st.metric("Feature Dimension", checkpoint["feature_dim"])
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        st.stop()

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
                    
                    # Step 1: Extract frames
                    status_text.text("Step 1/3: Extracting frames from video...")
                    progress_bar.progress(20)
                    
                    # Step 2: Load model (cached, so should be fast)
                    status_text.text("Step 2/3: Loading model...")
                    progress_bar.progress(40)
                    cnn, rnn, device, checkpoint = load_model()
                    
                    # Step 3: Run prediction
                    status_text.text("Step 3/3: Analyzing with deepfake detection model...")
                    progress_bar.progress(70)
                    
                    # Run prediction
                    prediction, confidence = predict_video(tmp_video_path, cnn, rnn, device)
                    
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

