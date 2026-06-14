from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse

import cv2

from common import LABELS, build_hands_detector, close_hands_detector, extract_hand_features, put_status


def read_existing_counts(csv_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {label: 0 for label in LABELS}
    if not csv_path.exists():
        return counts

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return counts
        for row in reader:
            if not row:
                continue
            label = row[0]
            if label in counts:
                counts[label] += 1
    return counts


def ensure_header(csv_path: Path) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        cols = ["label"] + [f"f{i}" for i in range(42)]
        writer.writerow(cols)


def append_sample(csv_path: Path, label: str, features) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([label, *features.tolist()])


def _resize_to_width(image, target_width: int):
    if target_width <= 0:
        return image
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / float(w)
    target_h = max(1, int(h * scale))
    return cv2.resize(image, (target_width, target_h), interpolation=cv2.INTER_AREA)


def _camera_url_candidates(camera_url: str):
    parsed = urlparse(camera_url)
    base_path = parsed.path or ""
    candidates = [camera_url]

    if base_path in {"", "/"}:
        for p in ["/video", "/videofeed", "/shot.jpg"]:
            candidates.append(urlunparse(parsed._replace(path=p, query="", fragment="")))
    elif base_path == "/video":
        candidates.append(urlunparse(parsed._replace(path="/videofeed", query="", fragment="")))
    elif base_path == "/videofeed":
        candidates.append(urlunparse(parsed._replace(path="/video", query="", fragment="")))

    out = []
    seen = set()
    for u in candidates:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _open_camera(camera_index: int, camera_url: Optional[str]):
    if camera_url:
        for url in _camera_url_candidates(camera_url):
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cap.release()
                continue

            # Validate by reading a few frames.
            ok = False
            for _ in range(20):
                ret, _ = cap.read()
                if ret:
                    ok = True
                    break

            if ok:
                return cap, url
            cap.release()
        return cv2.VideoCapture(), None

    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        return cap, None
    cap.release()
    return cv2.VideoCapture(), None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hand landmark samples from webcam")
    parser.add_argument("--output", default="data/landmarks.csv", help="Path to output CSV")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--camera-url", default=None, help="IP camera stream URL")
    parser.add_argument(
        "--backend",
        choices=["auto", "mediapipe_tasks", "mediapipe", "contour"],
        default="mediapipe",
        help="Hand feature extraction backend",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=800,
        help="Resize display window to this max width (0 disables resize)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_header(output_path)
    counts = read_existing_counts(output_path)

    hands = build_hands_detector(preferred_backend=args.backend)
    cap, used_url = _open_camera(args.camera_index, args.camera_url)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check --camera-url or --camera-index.")

    if used_url:
        print(f"Using camera URL: {used_url}")

    label_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            features, annotated = extract_hand_features(frame, hands)
            current_label = LABELS[label_idx]

            lines = [
                f"Mode: DATA COLLECTION",
                f"Current Label: {current_label}",
                f"Saved for label: {counts[current_label]}",
                "Keys: [c]=capture  [n/p]=next/prev label  [q]=quit",
            ]

            if features is None:
                lines.append("Hand: not detected")
                view = put_status(annotated, lines, color=(0, 0, 255))
            else:
                lines.append("Hand: detected")
                view = put_status(annotated, lines)

            cv2.imshow("Collect Landmarks", _resize_to_width(view, args.display_width))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                label_idx = (label_idx + 1) % len(LABELS)
            if key == ord("p"):
                label_idx = (label_idx - 1) % len(LABELS)
            if key == ord("c") and features is not None:
                append_sample(output_path, current_label, features.features)
                counts[current_label] += 1
                print(f"Saved sample for {current_label}. Total: {counts[current_label]}")
    finally:
        cap.release()
        close_hands_detector(hands)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
