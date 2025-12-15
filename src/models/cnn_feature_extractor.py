import torch
import torch.nn as nn
from torchvision import models

# ============================================================
# MobileNetV2 Feature Extractor (Pretrained)
# ============================================================

class MobileNetFeatureExtractor(nn.Module):
    """
    Pretrained MobileNetV2 encoder for feature extraction.
    Input: [B*T, 3, H, W] (batch of frames)
    Output: [B*T, 1280] feature vectors
    """
    def __init__(self):
        super(MobileNetFeatureExtractor, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features           # take convolutional backbone
        self.gap = nn.AdaptiveAvgPool2d((1, 1))     # global average pooling
        self.output_dim = 1280                       # MobileNetV2 output size

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)                   # flatten to [B*T, 1280]
        return x

# ============================================================
# Factory function
# ============================================================

def build_cnn():
    """
    Build MobileNetV2 feature extractor.
    Returns:
        model -> MobileNetFeatureExtractor instance
        feature_dim -> 1280 (output vector size)
    """
    m = MobileNetFeatureExtractor()
    return m, m.output_dim
