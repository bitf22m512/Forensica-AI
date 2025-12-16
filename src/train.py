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
# Save checkpoint (epoch-wise)
# ---------------------------------------------------------
def save_checkpoint(epoch, cnn, rnn, optimizer, best_val_acc, path):
    torch.save({
        "epoch": epoch,
        "cnn_state": cnn.state_dict(),
        "rnn_state": rnn.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc
    }, path)


# ---------------------------------------------------------
# Load checkpoint (resume training)
# ---------------------------------------------------------
def load_checkpoint(path, cnn, rnn, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    cnn.load_state_dict(checkpoint["cnn_state"])
    rnn.load_state_dict(checkpoint["rnn_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f">>> Resuming from epoch {checkpoint['epoch'] + 1}")
    return checkpoint["epoch"] + 1, checkpoint["best_val_acc"]


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

        seqs_reshaped = seqs.view(B * T, C, H, W)
        features = cnn(seqs_reshaped)
        feature_dim = features.shape[-1]
        features = features.view(B, T, feature_dim)

        logits = rnn(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += B

    return total_loss / total, correct / total


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

    return total_loss / total, correct / total


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def main():
    cfg = load_config("config.yaml")

    FRAMES_DIR = cfg["frames_dir"]
    LABELS_CSV = cfg["labels_csv"]
    SAVE_DIR = "models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = build_loaders(
        frames_root=FRAMES_DIR,
        labels_csv=LABELS_CSV,
        batch_size=cfg["batch_size"],
        train_split=cfg["train_split"],
        num_workers=2,
        num_frames=cfg["num_frames"],
        cache_in_memory=False
    )

    cnn, feature_dim = build_cnn()
    cnn.to(device)

    rnn = RNNClassifier(
        feature_dim=feature_dim,
        hidden_size=cfg["lstm_hidden_size"],
        num_layers=cfg["num_layers"],
        bidirectional=False
    )
    rnn.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(cnn.parameters()) + list(rnn.parameters()),
        lr=cfg["learning_rate"]
    )

    checkpoint_path = os.path.join(SAVE_DIR, "checkpoint.pth")
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
    final_model_path = os.path.join(SAVE_DIR, "final_model.pth")

    start_epoch = 0
    best_val_acc = 0.0

    if os.path.exists(checkpoint_path):
        start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path, cnn, rnn, optimizer, device
        )

    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{cfg['epochs']} ---")

        train_loss, train_acc = train_one_epoch(
            cnn, rnn, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            cnn, rnn, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        save_checkpoint(
            epoch, cnn, rnn, optimizer, best_val_acc, checkpoint_path
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "cnn_state": cnn.state_dict(),
                "rnn_state": rnn.state_dict(),
                "feature_dim": feature_dim
            }, best_model_path)
            print(f">>> Best model saved (acc={val_acc:.4f})")

    torch.save({
        "cnn_state": cnn.state_dict(),
        "rnn_state": rnn.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "feature_dim": feature_dim,
        "best_val_acc": best_val_acc,
        "epochs": cfg["epochs"],
        "config": cfg
    }, final_model_path)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
