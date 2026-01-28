# ============================================
# VIDEO DATASET CLASS
# ============================================
import os
import random
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

NUM_FRAMES = 20
FRAME_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

LABEL_MAP = {
    "real": 0,
    "fake": 1,
}

frame_transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

class VideoSequenceDataset(Dataset):
    """
    PyTorch Dataset returning a fixed-length sequence of frames for each video.
    """
    def __init__(self, frames_root: str, labels_csv: str, num_frames: int = NUM_FRAMES,
                 transform=frame_transform, shuffle_frames: bool = False, cache_in_memory: bool = False):
        self.frames_root = frames_root
        self.labels_df = pd.read_csv(labels_csv)
        self.num_frames = num_frames
        self.transform = transform
        self.shuffle_frames = shuffle_frames
        self.cache_in_memory = cache_in_memory

        self.samples: List[Tuple[str, int]] = []
        for _, row in self.labels_df.iterrows():
            video_filename = row["video"]
            video_folder = os.path.splitext(video_filename)[0]
            folder_path = os.path.join(self.frames_root, video_folder)
            if not os.path.isdir(folder_path):
                continue
            available = len([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))])
            if available < self.num_frames:
                continue
            label_str = str(row["label"]).lower()
            if label_str not in LABEL_MAP:
                continue
            label_int = LABEL_MAP[label_str]
            self.samples.append((video_folder, label_int))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found. Check frames_root and labels_csv paths.")

        self._cache = {} if self.cache_in_memory else None
        if self.cache_in_memory:
            print("Caching frames in memory (may use lots of RAM)...")
            for vid, _ in self.samples:
                folder = os.path.join(self.frames_root, vid)
                frames = self._read_frames_from_folder(folder)
                self._cache[vid] = frames

    def _read_frames_from_folder(self, folder: str) -> List[Image.Image]:
        """Return list of PIL images sorted by frame index."""
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))])
        files = files[:self.num_frames]
        images = []
        for fname in files:
            path = os.path.join(folder, fname)
            img = Image.open(path).convert("RGB")
            images.append(img)
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_folder, label = self.samples[idx]
        folder = os.path.join(self.frames_root, vid_folder)

        if self.cache_in_memory and vid_folder in self._cache:
            pil_frames = self._cache[vid_folder]
        else:
            pil_frames = self._read_frames_from_folder(folder)

        if self.shuffle_frames:
            pil_frames = pil_frames.copy()
            random.shuffle(pil_frames)

        frame_tensors = []
        for f in pil_frames:
            t = self.transform(f)
            frame_tensors.append(t)

        seq_tensor = torch.stack(frame_tensors, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return seq_tensor, label_tensor

def video_collate_fn(batch):
    """Collate function for video sequences."""
    seqs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    batch_seqs = torch.stack(seqs, dim=0)
    return batch_seqs, labels

def build_loaders(frames_root: str, labels_csv: str, batch_size: int = 4, train_split: float = 0.8,
                  num_workers: int = 2, **dataset_kwargs):
    dataset = VideoSequenceDataset(frames_root=frames_root, labels_csv=labels_csv, **dataset_kwargs)
    n = len(dataset)
    n_train = int(n * train_split)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=video_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=video_collate_fn, pin_memory=True)
    return train_loader, val_loader

print("✅ Dataset class defined!")
