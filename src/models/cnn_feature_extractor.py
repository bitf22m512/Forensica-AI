import torch
import torch.nn as nn
from torchvision import models

# ============================================================
# Custom CNN Feature Extractor (matches saved model)
# ============================================================

class CustomCNNFeatureExtractor(nn.Module):
    """
    Custom CNN encoder for feature extraction.
    Matches the architecture used during training.
    Input: [B*T, 3, H, W] (batch of frames)
    Output: [B*T, 256] feature vectors
    """
    def __init__(self):
        super(CustomCNNFeatureExtractor, self).__init__()
        # Convolutional layers matching saved model
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Fully connected layer matching saved model
        # fc input size: 65536 = 256 * 256, so spatial size needs to be 16x16 (256 = 16*16)
        # We'll use adaptive pooling to resize to 16x16 before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # 256 channels * 16*16 = 65536
        self.fc = nn.Linear(256 * 16 * 16, 256)
        
        self.relu = nn.ReLU()
        self.output_dim = 256

    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        # Adaptive pooling to ensure correct size for FC layer (16x16)
        x = self.adaptive_pool(x)  # -> [B*T, 256, 16, 16]
        x = x.view(x.size(0), -1)  # Flatten to [B*T, 256*16*16] = [B*T, 65536]
        x = self.fc(x)              # -> [B*T, 256]
        return x


# ============================================================
# MobileNetV2 Feature Extractor (Pretrained) - kept for compatibility
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

def build_cnn(use_custom=True):
    """
    Build CNN feature extractor.
    Args:
        use_custom: If True, use CustomCNNFeatureExtractor (matches saved model).
                    If False, use MobileNetV2.
    Returns:
        model -> Feature extractor instance
        feature_dim -> Output feature dimension
    """
    if use_custom:
        m = CustomCNNFeatureExtractor()
        return m, m.output_dim
    else:
        m = MobileNetFeatureExtractor()
        return m, m.output_dim
