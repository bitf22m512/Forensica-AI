import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from src.dataset.video_dataset import build_loaders
from src.models.cnn_feature_extractor import build_cnn
from src.models.rnn_classifier import RNNClassifier


MODEL_PATH = "models/best_model.pth"
FRAMES_DIR = "data/frames"
LABELS_CSV = "data/labels.csv"


def evaluate_model():

    # ------------------------
    # Device
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ------------------------
    # Load data (use only val_loader)
    # ------------------------
    _, val_loader = build_loaders(
        frames_root=FRAMES_DIR,
        labels_csv=LABELS_CSV,
        batch_size=4,
        train_split=0.8,
        num_workers=2,
        num_frames=20
    )

    # ------------------------
    # Load model
    # ------------------------
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    feature_dim = checkpoint["feature_dim"]

    cnn, _ = build_cnn("mobilenet")   # use same one used in training
    rnn = RNNClassifier(feature_dim=feature_dim)

    cnn.load_state_dict(checkpoint["cnn_state"])
    rnn.load_state_dict(checkpoint["rnn_state"])

    cnn.to(device)
    rnn.to(device)

    cnn.eval()
    rnn.eval()

    # ------------------------
    # Evaluation
    # ------------------------
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)

            B, T, C, H, W = seqs.shape
            seqs_reshaped = seqs.view(B*T, C, H, W)

            features = cnn(seqs_reshaped)
            features = features.view(B, T, -1)

            logits = rnn(features)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # ------------------------
    # Metrics
    # ------------------------
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Accuracy
    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"\nFinal Validation Accuracy: {acc:.4f}")


if __name__ == "__main__":
    evaluate_model()
