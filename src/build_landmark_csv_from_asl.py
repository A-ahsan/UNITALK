from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm

from common import LABELS, build_hands_detector, close_hands_detector, extract_hand_features


DATASET_LABEL_MAP = {
    "DELETE": "del",
}


def iter_image_files(folder: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        yield from folder.glob(ext)


def ensure_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        cols = ["label"] + [f"f{i}" for i in range(42)]
        writer.writerow(cols)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build landmark CSV from ASL image folders using MediaPipe"
    )
    parser.add_argument(
        "--dataset-dir",
        default=(
            "C:/Users/VIPER/.cache/kagglehub/datasets/grassknoted/asl-alphabet/"
            "versions/1/asl_alphabet_train/asl_alphabet_train"
        ),
        help="Path to ASL class folders (A, B, C, ...)",
    )
    parser.add_argument(
        "--output",
        default="data/landmarks_from_asl.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000,
        help="Max images to process per class (use -1 for all)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mediapipe_tasks", "mediapipe", "contour"],
        default="auto",
        help="Feature extraction backend",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    output_path = Path(args.output)
    ensure_header(output_path)

    hands = build_hands_detector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
        preferred_backend=args.backend,
    )
    print(f"Using backend: {hands.get('backend')}")

    total_seen = 0
    total_saved = 0

    try:
        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for label in LABELS:
                folder_name = DATASET_LABEL_MAP.get(label, label)
                class_dir = dataset_dir / folder_name
                if not class_dir.exists():
                    print(f"[SKIP] Class folder missing: {class_dir}")
                    continue

                images = sorted(iter_image_files(class_dir))
                if args.max_per_class >= 0:
                    images = images[: args.max_per_class]

                saved_for_class = 0

                for image_path in tqdm(images, desc=f"Processing {label}", unit="img"):
                    img = cv2.imread(str(image_path))
                    if img is None:
                        continue

                    total_seen += 1
                    features, _ = extract_hand_features(img, hands)
                    if features is None:
                        continue

                    writer.writerow([label, *features.features.tolist()])
                    saved_for_class += 1
                    total_saved += 1

                print(
                    f"[CLASS] {label}: seen={len(images)}, saved={saved_for_class}, "
                    f"dropped={len(images) - saved_for_class}"
                )
    finally:
        close_hands_detector(hands)

    print("\nDone.")
    print(f"Images processed: {total_seen}")
    print(f"Rows saved: {total_saved}")
    print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()
