import cv2
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from src.models.cnn_feature_extractor import build_cnn
from src.models.rnn_classifier import RNNClassifier

# Settings
MODEL_PATH = "models/best_model.pth"
FRAME_SIZE = 128
NUM_FRAMES = 20
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


# -------------------------------------------------
# Extract frames from a single video (for inference)
# -------------------------------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < NUM_FRAMES:
        raise ValueError("Video too short for inference!")

    interval = total // NUM_FRAMES
    frames = []

    for i in range(NUM_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    cap.release()
    return frames


# -------------------------------------------------
# Inference function
# -------------------------------------------------
def predict(video_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    feature_dim = checkpoint["feature_dim"]

    cnn, _ = build_cnn("mobilenet")
    rnn = RNNClassifier(feature_dim=feature_dim)

    cnn.load_state_dict(checkpoint["cnn_state"])
    rnn.load_state_dict(checkpoint["rnn_state"])

    cnn.to(device)
    rnn.to(device)

    cnn.eval()
    rnn.eval()

    # Extract frames
    frames = extract_frames(video_path)

    # Preprocess
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

    print("\n----------------------------")
    print("Video:", video_path)
    print("Prediction:", label_map[pred_class])
    print(f"Confidence: {confidence:.4f}")
    print("----------------------------\n")

    return label_map[pred_class], confidence


if __name__ == "__main__":
    video = input("Enter path to video: ")
    predict(video)
