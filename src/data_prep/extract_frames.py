# ============================================
# EXTRACT FRAMES FROM ALL VIDEOS (Kaggle-ready)
# ============================================
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_frames_from_video(video_path, output_dir, num_frames=20, frame_size=128):
    if not video_path.exists():
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames + 10:
        cap.release()
        return False

    start_frame = int(0.1 * total_frames)
    usable_frames = total_frames - start_frame
    interval = max(1, usable_frames // num_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    for i in range(num_frames):
        frame_idx = start_frame + i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take the first detected face
            frame = frame[y:y+h, x:x+w]

        # Resize and save
        frame = cv2.resize(frame, (frame_size, frame_size))
        out_path = output_dir / f"frame_{i}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        extracted += 1

    cap.release()
    return extracted == num_frames


def extract_all_frames(labels_csv="data/labels.csv",
                       dataset_root="/kaggle/input/celeb-df-v2",
                       frames_dir="data/frames",
                       num_frames=20,
                       frame_size=128):
    labels = pd.read_csv(labels_csv)
    dataset_root = Path(dataset_root)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    success_count, skip_count = 0, 0

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Extracting frames"):
        video_name = row["video"]

        # Look in all possible subfolders
        candidate_paths = [
            dataset_root / "Celeb-real" / video_name,
            dataset_root / "Celeb-synthesis" / video_name,
            dataset_root / "YouTube-real" / video_name
        ]
        video_path = next((p for p in candidate_paths if p.exists()), None)
        if video_path is None:
            skip_count += 1
            continue

        output_dir = frames_dir / video_path.stem
        ok = extract_frames_from_video(video_path, output_dir, num_frames=num_frames, frame_size=frame_size)
        if ok:
            success_count += 1
        else:
            skip_count += 1

    print("\n✅ Frame extraction completed")
    print(f"   Success: {success_count}")
    print(f"   Skipped: {skip_count}")
    return success_count, skip_count

# -----------------------------
# Run full extraction
# -----------------------------                                                                                                                                                                                                                                                                                                        
success, skipped = extract_all_frames(
    labels_csv="data/labels.csv",
    dataset_root="/kaggle/input/celeb-df-v2",
    frames_dir="data/frames",
    num_frames=20,
    frame_size=128
)
