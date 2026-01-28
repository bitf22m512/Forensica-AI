# ============================================
# INFERENCE ON SINGLE VIDEO (CNN + RNN + MCTNN)
# ============================================
from torchvision import transforms
from PIL import Image
import torch
import cv2
import os

def extract_frames_inference(video_path, num_frames=20):
    """Extract frames from a single video for inference."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total < num_frames:
        raise ValueError("Video too short for inference!")
    
    interval = total // num_frames
    frames = []
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)
    
    cap.release()
    return frames

def predict_video(video_path, model_path="models/best_model.pth"):
    """Predict if a video is real or fake using CNN + RNN + MCTNN."""
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the model first.")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    feature_dim = checkpoint["feature_dim"]
    
    # Rebuild models using the correct classes
    cnn = MobileNetFeatureExtractor(freeze_backbone=False)
    rnn = RNNClassifier(
        feature_dim=feature_dim,
        hidden_size=checkpoint["rnn_params"]["hidden_size"],
        num_layers=checkpoint["rnn_params"]["num_layers"],
        bidirectional=checkpoint["rnn_params"]["bidirectional"]
    )
    mctnn = MCTNN(feature_dim=rnn.output_dim)
    
    # Load saved states
    cnn.load_state_dict(checkpoint["cnn_state"])
    rnn.load_state_dict(checkpoint["rnn_state"])
    mctnn.load_state_dict(checkpoint["mctnn_state"])
    
    cnn.to(device)
    rnn.to(device)
    mctnn.to(device)
    
    cnn.eval()
    rnn.eval()
    mctnn.eval()
    
    # Extract frames
    frames = extract_frames_inference(video_path)
    
    # Preprocess frames
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    tensors = [transform(f) for f in frames]
    seq_tensor = torch.stack(tensors, dim=0)   # [T, C, H, W]
    seq_tensor = seq_tensor.unsqueeze(0)       # [1, T, C, H, W]
    
    # Forward pass
    with torch.no_grad():
        B, T, C, H, W = seq_tensor.shape
        seq_tensor = seq_tensor.to(device)
        
        # CNN features
        reshaped = seq_tensor.view(B*T, C, H, W)
        features = cnn(reshaped)
        features = features.view(B, T, -1)
        
        # RNN temporal encoding
        rnn_out = rnn(features)
        
        # Expand for MCTNN
        rnn_seq = rnn_out.unsqueeze(1).repeat(1, T, 1)
        
        # MCTNN logits
        logits = mctnn(rnn_seq)
        probs = torch.softmax(logits, dim=1)
    
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    
    label_map = {0: "REAL", 1: "FAKE"}
    
    print("\n" + "="*40)
    print("🎬 Video Prediction")
    print("="*40)
    print(f"Video: {video_path}")
    print(f"Prediction: {label_map[pred_class]}")
    print(f"Confidence: {confidence:.4f}")
    print("="*40)
    
    return label_map[pred_class], confidence

# Example usage:
predict_video('/kaggle/input/celeb-df-v2/Celeb-real/id0_0001.mp4')

print("✅ Inference function ready for CNN + RNN + MCTNN!")
