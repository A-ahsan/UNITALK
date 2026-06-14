from __future__ import annotations

import argparse
from pathlib import Path

import cv2

CLASS_MAP = {
    "del": "DELETE",
    "nothing": "NOTHING",
    "space": "SPACE",
}


def label_from_folder(name: str) -> str:
    return CLASS_MAP.get(name, name.upper())


def collect_images(dataset_dir: Path):
    class_dirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    items = []
    for cdir in class_dirs:
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            images.extend(cdir.glob(ext))
        images = sorted(images)
        if images:
            items.append((cdir.name, images))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse ASL dataset images by class")
    parser.add_argument(
        "--dataset-dir",
        default=(
            "C:/Users/VIPER/.cache/kagglehub/datasets/grassknoted/asl-alphabet/"
            "versions/1/asl_alphabet_train/asl_alphabet_train"
        ),
    )
    parser.add_argument("--window", default="ASL Dataset Viewer")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    classes = collect_images(dataset_dir)
    if not classes:
        raise RuntimeError("No class images found")

    class_idx = 0
    image_idx = 0

    print("Controls:")
    print("  n = next class, p = previous class")
    print("  k = next image, j = previous image")
    print("  q = quit")

    while True:
        folder, images = classes[class_idx]
        if image_idx >= len(images):
            image_idx = 0

        path = images[image_idx]
        img = cv2.imread(str(path))
        if img is None:
            image_idx = (image_idx + 1) % len(images)
            continue

        label = label_from_folder(folder)
        text1 = f"Label: {label}"
        text2 = f"Class {class_idx + 1}/{len(classes)} | Image {image_idx + 1}/{len(images)}"
        text3 = "n/p class, k/j image, q quit"

        view = img.copy()
        cv2.rectangle(view, (10, 10), (650, 100), (0, 0, 0), -1)
        cv2.putText(view, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, text2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, text3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2, cv2.LINE_AA)

        cv2.imshow(args.window, view)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("n"):
            class_idx = (class_idx + 1) % len(classes)
            image_idx = 0
        elif key == ord("p"):
            class_idx = (class_idx - 1) % len(classes)
            image_idx = 0
        elif key == ord("k"):
            image_idx = (image_idx + 1) % len(images)
        elif key == ord("j"):
            image_idx = (image_idx - 1) % len(images)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
