import cv2
import os
import logging
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file at {config_path}")
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def extract_frames_from_video(video_path: Path, output_dir: Path, num_frames: int, frame_size: int) -> bool:
    """
    Extract evenly spaced frames from a video file.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        frame_size: Target size for resizing frames (width, height)
    
    Returns:
        True if all frames were successfully extracted, False otherwise
    """
    # Validate video file exists
    if not video_path.is_file():
        logger.warning(f"Video file not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Check if video opened successfully
    if not cap.isOpened():
        logger.warning(f"Failed to open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        logger.warning(f"Video too short: {video_path} ({total_frames} frames < {num_frames} required)")
        cap.release()
        return False

    # Calculate frame interval for even distribution
    interval = max(1, total_frames // num_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    
    for i in range(num_frames):
        frame_index = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {i} from {video_path}")
            continue

        # Resize frame
        frame = cv2.resize(frame, (frame_size, frame_size))
        out_path = output_dir / f"frame_{i}.jpg"
        
        # Save frame with error handling
        if not cv2.imwrite(str(out_path), frame):
            logger.error(f"Failed to write frame {i} to {out_path}")
            continue
        
        extracted_count += 1

    cap.release()
    
    # Verify we extracted the expected number of frames
    if extracted_count < num_frames:
        logger.warning(f"Only extracted {extracted_count}/{num_frames} frames from {video_path}")
        return False
    
    return True


def main():
    # Get config path relative to script location
    script_dir = Path(__file__).parent.parent.parent
    config_path = script_dir / "config.yaml"
    
    cfg = load_config(config_path)
    num_frames = int(cfg.get("num_frames", 20))
    frame_size = int(cfg.get("frame_size", 128))
    raw_video_dir = Path(cfg.get("raw_video_dir", "data/raw_videos"))
    output_frame_dir = Path(cfg.get("frames_dir", "data/frames"))
    labels_csv = Path(cfg.get("labels_csv", "data/labels.csv"))
    
    # Resolve paths relative to project root
    if not raw_video_dir.is_absolute():
        raw_video_dir = script_dir / raw_video_dir
    if not output_frame_dir.is_absolute():
        output_frame_dir = script_dir / output_frame_dir
    if not labels_csv.is_absolute():
        labels_csv = script_dir / labels_csv

    if not labels_csv.is_file():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}. Run prepare_celeb_df_v2 first.")
    
    if not raw_video_dir.is_dir():
        raise FileNotFoundError(f"Raw video directory not found: {raw_video_dir}")

    labels = pd.read_csv(labels_csv)
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting frame extraction for {len(labels)} videos...")
    logger.info(f"Configuration: {num_frames} frames, {frame_size}x{frame_size} resolution")

    success_count = 0
    skip_count = 0
    
    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Extracting frames"):
        filename = row["video"]
        video_path = raw_video_dir / filename

        if not video_path.is_file():
            logger.warning(f"Video file not found: {video_path}")
            skip_count += 1
            continue

        out_dir = output_frame_dir / Path(filename).stem
        success = extract_frames_from_video(video_path, out_dir, num_frames, frame_size)
        
        if success:
            success_count += 1
        else:
            skip_count += 1
            logger.debug(f"Skipping video: {filename}")

    logger.info(f"Frame extraction complete. Success: {success_count}, Skipped: {skip_count}, Total: {len(labels)}")
    print(f"\n✅ Frame extraction complete!")
    print(f"   Successfully processed: {success_count} videos")
    print(f"   Skipped: {skip_count} videos")


if __name__ == "__main__":
    main()
