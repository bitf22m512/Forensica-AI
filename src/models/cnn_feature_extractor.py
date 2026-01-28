# ============================================
# CNN + MCTNN FEATURE EXTRACTOR (Kaggle-ready, no MTCNN)
# ============================================
import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange

# -------------------------------
# MobileNetV2 feature extractor
# -------------------------------
class MobileNetFeatureExtractor(nn.Module):
    """
    MobileNetV2-based spatial feature extractor
    Input:  [B*T, 3, H, W]
    Output: [B*T, 256]
    """
    def __init__(self, freeze_backbone=True):
        super().__init__()

        # Kaggle offline: no pretrained weights
        mobilenet = models.mobilenet_v2(weights=None)

        # Backbone
        self.backbone = mobilenet.features

        # Freeze early layers if desired
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.output_dim = 256

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -------------------------------
# Multi-Channel Temporal Neural Network (MCTNN)
# -------------------------------
class MCTNN(nn.Module):
    """
    MCTNN for temporal modeling
    Input: [B, T, feature_dim] (e.g., RNN output expanded over time)
    Output: [B, num_classes] (real/fake logits)
    """
    def __init__(self, feature_dim=128, hidden_dim=128, kernel_sizes=[3,5,7], dropout=0.3, num_classes=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Temporal convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            for k in kernel_sizes
        ])

        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * len(kernel_sizes), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: [B, T, feature_dim]
        """
        # Convert to [B, feature_dim, T] for Conv1d
        x = rearrange(x, "b t f -> b f t")

        conv_outs = [conv(x) for conv in self.conv_layers]

        # Global max pooling over temporal dimension
        pooled = [torch.max(o, dim=2)[0] for o in conv_outs]

        out = torch.cat(pooled, dim=1)
        logits = self.fc(out)
        return logits


# -------------------------------
# Build CNN + optional MCTNN
# -------------------------------
def build_cnn_mctnn(cnn_freeze=True, mctnn_input_dim=128, use_mctnn=True):
    """
    Returns:
        cnn_model: MobileNet feature extractor
        mctnn_model: MCTNN classifier (optional)
        cnn_output_dim: output dimension of CNN
    """
    cnn_model = MobileNetFeatureExtractor(freeze_backbone=cnn_freeze)
    
    if use_mctnn:
        mctnn_model = MCTNN(feature_dim=mctnn_input_dim)
    else:
        mctnn_model = None

    return cnn_model, mctnn_model, cnn_model.output_dim


print("✅ CNN + MCTNN block ready for Kaggle (offline)")
print("   - CNN output_dim:", MobileNetFeatureExtractor().output_dim)
print("   - MCTNN input_dim should match your RNN output_dim")
