import cv2
import os
import pandas as pd
from tqdm import tqdm

# CONFIG
NUM_FRAMES = 20
FRAME_SIZE = 128

RAW_VIDEO_DIR = "data/raw_videos/"
OUTPUT_FRAME_DIR = "data/frames/"
LABEL_FILE = "data/labels.csv"

os.makedirs(OUTPUT_FRAME_DIR, exist_ok=True)

def extract_frames_from_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Skip videos with too few frames
    if total_frames < NUM_FRAMES:
        return False

    # Pick frame intervals
    interval = total_frames // NUM_FRAMES
    frames = []

    for i in range(NUM_FRAMES):
        frame_index = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        out_path = os.path.join(output_dir, f"frame_{i}.jpg")
        cv2.imwrite(out_path, frame)
        frames.append(out_path)

    cap.release()
    return True


def main():
    labels = pd.read_csv(LABEL_FILE)

    for idx, row in tqdm(labels.iterrows(), total=len(labels)):
        filename = row["video"]
        video_path = os.path.join(RAW_VIDEO_DIR, filename)

        out_dir = os.path.join(OUTPUT_FRAME_DIR, filename.replace('.mp4', ''))
        os.makedirs(out_dir, exist_ok=True)

        success = extract_frames_from_video(video_path, out_dir)
        if not success:
            print(f"Skipping video (too short): {filename}")

    print("Frame extraction complete.")

if __name__ == "__main__":
    main()
