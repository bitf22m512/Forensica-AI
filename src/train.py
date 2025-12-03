import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import os

from src.dataset.video_dataset import build_loaders
from src.models.cnn_feature_extractor import build_cnn
from src.models.rnn_classifier import RNNClassifier


# ---------------------------------------------------------
# Load config.yaml
# ---------------------------------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# Train for one epoch
# ---------------------------------------------------------
def train_one_epoch(cnn, rnn, loader, criterion, optimizer, device):
    cnn.train()
    rnn.train()

    total_loss = 0
    correct = 0
    total = 0

    for seqs, labels in tqdm(loader, desc="Training", ncols=100):
        seqs = seqs.to(device)       # [B, T, C, H, W]
        labels = labels.to(device)   # [B]

        B, T, C, H, W = seqs.shape

        # Merge batch & time to feed CNN
        seqs_reshaped = seqs.view(B * T, C, H, W)

        features = cnn(seqs_reshaped)               # [B*T, feature_dim]
        feature_dim = features.shape[-1]
        features = features.view(B, T, feature_dim) # [B, T, feature_dim]

        logits = rnn(features)   # [B, 2]

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += B

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


# ---------------------------------------------------------
# Validation loop
# ---------------------------------------------------------
def validate(cnn, rnn, loader, criterion, device):
    cnn.eval()
    rnn.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for seqs, labels in tqdm(loader, desc="Validating", ncols=100):
            seqs = seqs.to(device)
            labels = labels.to(device)

            B, T, C, H, W = seqs.shape

            seqs_reshaped = seqs.view(B * T, C, H, W)
            features = cnn(seqs_reshaped)
            feature_dim = features.shape[-1]
            features = features.view(B, T, feature_dim)

            logits = rnn(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * B

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += B

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def main():
    cfg = load_config("config.yaml")

    # Directories
    FRAMES_DIR = "data/frames"
    LABELS_CSV = "data/labels.csv"
    SAVE_DIR = "models/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build dataloaders
    train_loader, val_loader = build_loaders(
        frames_root=FRAMES_DIR,
        labels_csv=LABELS_CSV,
        batch_size=cfg["batch_size"],
        train_split=cfg["train_split"],
        num_workers=2,
        num_frames=cfg["num_frames"],
        cache_in_memory=False
    )

    # Build CNN
    cnn, feature_dim = build_cnn(
        model_name="mobilenet",     # OR "simple"
        output_dim=cfg["cnn_output_dim"]
    )
    cnn.to(device)

    # Build RNN
    rnn = RNNClassifier(
        feature_dim=feature_dim,
        hidden_size=cfg["lstm_hidden_size"],
        num_layers=cfg["num_layers"],
        bidirectional=False
    )
    rnn.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(cnn.parameters()) + list(rnn.parameters()),
        lr=cfg["learning_rate"]
    )

    # Training loop
    best_val_acc = 0

    for epoch in range(cfg["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{cfg['epochs']} ---")

        train_loss, train_acc = train_one_epoch(
            cnn, rnn, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            cnn, rnn, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save({
                "cnn_state": cnn.state_dict(),
                "rnn_state": rnn.state_dict(),
                "feature_dim": feature_dim
            }, save_path)
            print(f">>> Saved new best model at acc={val_acc:.4f}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
