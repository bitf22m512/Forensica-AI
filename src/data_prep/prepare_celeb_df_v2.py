import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Iterable, Tuple


REAL_DIRNAME = "Celeb-real"
FAKE_DIRNAME = "Celeb-synthesis"


def iter_videos(root: Path) -> Iterable[Tuple[Path, str]]:
    """Yield (video_path, label_str) for all mp4s in Celeb-DF-v2 layout."""
    real_dir = root / REAL_DIRNAME
    fake_dir = root / FAKE_DIRNAME
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise FileNotFoundError(
            f"Expected subfolders '{REAL_DIRNAME}' and '{FAKE_DIRNAME}' under {root}"
        )

    for path in sorted(real_dir.glob("*.mp4")):
        yield path, "real"
    for path in sorted(fake_dir.glob("*.mp4")):
        yield path, "fake"


def maybe_limit_per_class(
    items: Iterable[Tuple[Path, str]], limit: int | None
) -> list[Tuple[Path, str]]:
    if limit is None:
        return list(items)
    kept: list[Tuple[Path, str]] = []
    counts = {"real": 0, "fake": 0}
    for path, label in items:
        if counts[label] < limit:
            kept.append((path, label))
            counts[label] += 1
    return kept


def copy_videos(samples: Iterable[Tuple[Path, str]], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src, label in samples:
        dest = dest_dir / src.name
        if dest.exists():
            continue
        shutil.copy2(src, dest)


def write_labels(samples: Iterable[Tuple[Path, str]], labels_csv: Path) -> None:
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with labels_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "label"])
        for src, label in samples:
            writer.writerow([src.name, label])


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Celeb-DF-v2: copy videos into data/raw_videos and write labels.csv"
    )
    parser.add_argument(
        "--celeb_root",
        type=Path,
        default=Path("Celeb-DF-v2"),
        help="Path to Celeb-DF-v2 directory (containing Celeb-real and Celeb-synthesis)",
    )
    parser.add_argument(
        "--output_raw",
        type=Path,
        default=Path("data/raw_videos"),
        help="Destination directory for raw videos",
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        default=Path("data/labels.csv"),
        help="Where to write the labels file (video,label)",
    )
    parser.add_argument(
        "--limit_per_class",
        type=int,
        default=None,
        help="Optional cap per class for quick experiments (e.g., 500)",
    )
    args = parser.parse_args()

    all_samples = list(iter_videos(args.celeb_root))
    samples = maybe_limit_per_class(all_samples, args.limit_per_class)

    print(f"Found {len(all_samples)} videos (real+fake); using {len(samples)}.")
    copy_videos(samples, args.output_raw)
    write_labels(samples, args.labels_csv)
    print(f"Wrote labels to {args.labels_csv} and videos to {args.output_raw}.")


if __name__ == "__main__":
    main()

