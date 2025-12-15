import pandas as pd
from pathlib import Path

# Read full labels
labels_df = pd.read_csv("data/labels.csv")

# Get all video folders that exist in data/frames
frames_dir = Path("data/frames")
existing_folders = {f.name for f in frames_dir.iterdir() if f.is_dir()}

# Filter labels to only videos that have frames extracted
# Remove .mp4 extension to match folder names
labels_df["video_name"] = labels_df["video"].apply(lambda x: Path(x).stem)
subset_df = labels_df[labels_df["video_name"].isin(existing_folders)]

# Save subset (keep original columns)
subset_df[["video", "label"]].to_csv("data/labels_subset.csv", index=False)

print(f"Original videos: {len(labels_df)}")
print(f"Videos with frames: {len(subset_df)}")
print(f"Saved to data/labels_subset.csv")

