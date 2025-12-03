import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# OPTION 1 — Custom CNN (Simple & Fast)
# Output: feature vector of size cnn_output_dim
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, output_dim=256):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)   # -> 32x128x128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # -> 64x64x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # -> 128x32x32
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)# -> 256x16x16

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected projection to output_dim
        self.fc = nn.Linear(256 * 16 * 16, output_dim)

    def forward(self, x):
        """
        Input: x shape [B*T, 3, 128, 128]
        Output: feature vector [B*T, output_dim]
        """

        x = self.pool(F.relu(self.conv1(x)))  # -> 32x64x64
        x = self.pool(F.relu(self.conv2(x)))  # -> 64x32x32
        x = self.pool(F.relu(self.conv3(x)))  # -> 128x16x16
        x = F.relu(self.conv4(x))             # -> 256x16x16 (no pool)

        x = x.view(x.size(0), -1)             # flatten
        x = self.fc(x)                        # project to feature vector

        return x


# ============================================================
# OPTION 2 — Pretrained MobileNetV2 Encoder (Recommended)
# Uses ImageNet pretrained weights and outputs 1280-d vector
# ============================================================

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetFeatureExtractor, self).__init__()

        mobilenet = models.mobilenet_v2(pretrained=True)

        # Take the feature extractor only
        self.features = mobilenet.features

        # Global average pooling gives feature vector
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.output_dim = 1280  # MobileNetV2 output size

    def forward(self, x):
        """
        Input: [B*T, 3, H, W]
        Output: [B*T, 1280]
        """
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


# ============================================================
# Factory function (select model in train.py)
# ============================================================

def build_cnn(model_name="simple", output_dim=256):
    """
    Options:
        model_name="simple"          → SimpleCNN(output_dim)
        model_name="mobilenet"       → MobileNetFeatureExtractor()
    """
    if model_name == "simple":
        return SimpleCNN(output_dim=output_dim), output_dim

    elif model_name == "mobilenet":
        m = MobileNetFeatureExtractor()
        return m, m.output_dim

    else:
        raise ValueError("Unknown CNN model name")
