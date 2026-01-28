# ============================================
# EVALUATION (Kaggle-ready)
# ============================================
import torch
from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = os.path.join(CONFIG["save_dir"], "best_model.pth")

if os.path.exists(MODEL_PATH):
    print("📊 Evaluating model...")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=CONFIG["device"])
    feature_dim = checkpoint["feature_dim"]

    # Rebuild models
    cnn, mctnn, cnn_output_dim = build_cnn_mctnn(
        cnn_freeze=CONFIG["cnn_freeze_backbone"],
        mctnn_input_dim=CONFIG["mctnn_hidden_dim"],
        use_mctnn=True
    )
    rnn = RNNClassifier(
        feature_dim=cnn_output_dim,
        hidden_size=CONFIG["rnn_hidden_size"],
        num_layers=CONFIG["rnn_num_layers"],
        bidirectional=False
    )

    # Load weights
    cnn.load_state_dict(checkpoint["cnn_state"])
    rnn.load_state_dict(checkpoint["rnn_state"])
    if mctnn:
        mctnn.load_state_dict(checkpoint["mctnn_state"])

    cnn.to(CONFIG["device"]).eval()
    rnn.to(CONFIG["device"]).eval()
    if mctnn:
        mctnn.to(CONFIG["device"]).eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for seqs, labels in tqdm(val_loader, desc="Evaluating", ncols=100):
            seqs = seqs.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])

            B, T, C, H, W = seqs.shape
            seqs_reshaped = seqs.view(B * T, C, H, W)

            features = cnn(seqs_reshaped)
            features = features.view(B, T, -1)

            rnn_out = rnn(features)
            rnn_seq = rnn_out.unsqueeze(1).repeat(1, T, 1)

            logits = mctnn(rnn_seq)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    acc = (torch.tensor(all_labels) == torch.tensor(all_preds)).float().mean().item()
    print(f"\n✅ Final Validation Accuracy: {acc:.4f}")

else:
    print("❌ Model not found. Please train the model first.")
