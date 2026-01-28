# ============================================
# MAIN TRAINING LOOP – CNN + RNN + MCTNN
# ============================================
import torch
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Build data loaders (assume build_loaders is implemented)
train_loader, val_loader = build_loaders(
    frames_root=CONFIG["frames_dir"],
    labels_csv=CONFIG["labels_csv"],
    batch_size=CONFIG["batch_size"],
    train_split=CONFIG["train_split"],
    num_workers=2,
    num_frames=CONFIG["num_frames"],
    cache_in_memory=False
)
print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

# Build models (CNN, RNN, MCTNN)
cnn_model = MobileNetFeatureExtractor(freeze_backbone=CONFIG["cnn_freeze_backbone"])
cnn_model.to(device)

rnn_model = RNNClassifier(
    feature_dim=cnn_model.output_dim,
    hidden_size=CONFIG["rnn_hidden_size"],
    num_layers=CONFIG["rnn_num_layers"]
)
rnn_model.to(device)

mctnn_model = MCTNN(feature_dim=CONFIG["rnn_hidden_size"])
mctnn_model.to(device)

# Weighted loss to handle imbalance (REAL weighted higher)
class_weights = torch.tensor([5.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    list(cnn_model.parameters()) + list(rnn_model.parameters()) + list(mctnn_model.parameters()),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)

# Training loop
best_val_acc = 0.0
os.makedirs(CONFIG["save_dir"], exist_ok=True)

for epoch in range(CONFIG["epochs"]):
    print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
    train_loss, train_acc = train_one_epoch(cnn_model, rnn_model, mctnn_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(cnn_model, rnn_model, mctnn_model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "cnn_state": cnn_model.state_dict(),
            "rnn_state": rnn_model.state_dict(),
            "mctnn_state": mctnn_model.state_dict(),
            "feature_dim": cnn_model.output_dim,
            "config": CONFIG,
            "best_val_acc": best_val_acc,
            "epoch": epoch+1
        }, os.path.join(CONFIG["save_dir"], "best_model.pth"))
        print(f"✅ Best model saved at epoch {epoch+1}")
