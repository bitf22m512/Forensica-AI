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
# Load checkpoint (resume training)
# ---------------------------------------------------------
def load_checkpoint(path, cnn, rnn, optimizer, device):
    """Load checkpoint and resume training from saved state."""
    checkpoint = torch.load(path, map_location=device)
    cnn.load_state_dict(checkpoint["cnn_state"])
    rnn.load_state_dict(checkpoint["rnn_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    print(f">>> Resuming from epoch {start_epoch}")
    print(f">>> Previous best validation accuracy: {best_val_acc:.4f}")
    return start_epoch, best_val_acc


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
        num_workers=2,  # Set to 2 for faster data loading
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

    # RNN parameters for saving/loading
    rnn_params = {
        "feature_dim": feature_dim,
        "hidden_size": cfg["lstm_hidden_size"],
        "num_layers": cfg["num_layers"],
        "bidirectional": False
    }

    # Try to resume from checkpoint
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path, cnn, rnn, optimizer, device
        )
        # Load config from checkpoint if available
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        if "config" in checkpoint_data:
            print(">>> Loaded config from checkpoint")
        if "rnn_params" in checkpoint_data:
            rnn_params = checkpoint_data["rnn_params"]
            print(f">>> Loaded RNN params: {rnn_params}")
    elif os.path.exists(best_model_path):
        # If no checkpoint but best model exists, try to resume from best model
        print(">>> No checkpoint found, attempting to resume from best_model.pth")
        try:
            best_model_data = torch.load(best_model_path, map_location=device)
            cnn.load_state_dict(best_model_data["cnn_state"])
            rnn.load_state_dict(best_model_data["rnn_state"])
            
            # Try to get epoch info, default to 0 if not found (start from beginning)
            start_epoch = best_model_data.get("epoch", 0)
            best_val_acc = best_model_data.get("best_val_acc", 0.0)
            
            # Update rnn_params if available
            if "rnn_params" in best_model_data:
                rnn_params = best_model_data["rnn_params"]
            elif "feature_dim" in best_model_data:
                # Reconstruct rnn_params from saved feature_dim
                rnn_params["feature_dim"] = best_model_data["feature_dim"]
            
            print(f">>> Resumed from best_model.pth")
            if "epoch" in best_model_data:
                print(f">>> Previous epoch: {start_epoch}, resuming from epoch {start_epoch + 1}")
            else:
                print(f">>> No epoch info found, starting from epoch 1 (will train all {cfg['epochs']} epochs)")
                start_epoch = 0  # Start from beginning to train all epochs
            if best_val_acc > 0:
                print(f">>> Previous best validation accuracy: {best_val_acc:.4f}")
            # Note: optimizer state not available, will start fresh
            print(">>> Note: Optimizer state not available, starting with fresh optimizer state")
        except Exception as e:
            print(f">>> Could not load from best_model.pth: {e}")
            print(">>> Starting training from scratch")
            start_epoch = 0
            best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch + 1}/{cfg['epochs']}")
    print(f"Best validation accuracy so far: {best_val_acc:.4f}")
    print(f"{'='*60}\n")

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

        # Save checkpoint after each epoch (latest checkpoint)
        checkpoint_data = {
            "epoch": epoch,
            "cnn_state": cnn.state_dict(),
            "rnn_state": rnn.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "feature_dim": feature_dim,
            "rnn_params": rnn_params,
            "config": cfg,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f">>> Checkpoint saved: {checkpoint_path}")

        # Also save epoch-specific checkpoint
        epoch_checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint_data, epoch_checkpoint_path)
        print(f">>> Epoch checkpoint saved: {epoch_checkpoint_path}")

        # Update best model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model with all necessary info for transfer
            best_model_data = {
                "cnn_state": cnn.state_dict(),
                "rnn_state": rnn.state_dict(),
                "feature_dim": feature_dim,
                "rnn_params": rnn_params,
                "config": cfg,
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            torch.save(best_model_data, best_model_path)
            print(f">>> Best model saved (acc={val_acc:.4f}) at epoch {epoch + 1}")

    # Save final model with complete information
    final_model_data = {
        "cnn_state": cnn.state_dict(),
        "rnn_state": rnn.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "feature_dim": feature_dim,
        "rnn_params": rnn_params,
        "best_val_acc": best_val_acc,
        "epochs": cfg["epochs"],
        "config": cfg,
        "final_epoch": cfg["epochs"]
    }
    torch.save(final_model_data, final_model_path)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Latest checkpoint: {checkpoint_path}")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print("="*60)
    print("\nAll models are saved with complete information and can be transferred to other machines.")


if __name__ == "__main__":
    main()
