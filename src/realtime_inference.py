from __future__ import annotations

import argparse
import collections
import os
import time
from pathlib import Path
from statistics import mode, StatisticsError
from typing import Optional
from urllib.parse import urlparse, urlunparse
import warnings

# Prefer X11 backend and a system font path for OpenCV HighGUI on Raspberry Pi.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import joblib
import numpy as np
import pandas as pd

from common import (
    build_hands_detector,
    close_hands_detector,
    extract_hand_features,
    preprocess_frame,
    put_status,
)

# Suppress noisy protobuf deprecation spam from mediapipe dependencies.
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)


DEFAULT_MODEL_CANDIDATES = [
    "models/sign_model_tasks_v6_combo.joblib",
    "models/sign_model_tasks_v5_fresh.joblib",
    "models/sign_model_tasks_v4.joblib",
    "models/sign_model_tasks_v2.joblib",
    "models/sign_model_tasks.joblib",
    "models/sign_model.joblib",
]


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with train_classifier.py"
        )
    data = joblib.load(model_path)
    if "model" not in data:
        raise ValueError("Invalid model file format")
    model = data["model"]
    expected_feature_count = data.get("feature_count")
    if expected_feature_count is None and hasattr(model, "n_features_in_"):
        expected_feature_count = int(model.n_features_in_)
    return model, expected_feature_count


def resolve_model_path(explicit_model: Optional[str]) -> Path:
    if explicit_model:
        return Path(explicit_model)

    for candidate in DEFAULT_MODEL_CANDIDATES:
        p = Path(candidate)
        if p.exists() and p.is_file():
            return p

    return Path(DEFAULT_MODEL_CANDIDATES[0])


def _build_inference_frame(clf, features_vector, expected_feature_count: Optional[int]):
    vec = np.asarray(features_vector, dtype=np.float32).reshape(-1)

    if expected_feature_count is not None and len(vec) != int(expected_feature_count):
        return None, (
            f"Model expects {int(expected_feature_count)} features, "
            f"but detector produced {len(vec)}."
        )

    if hasattr(clf, "feature_names_in_"):
        cols = list(clf.feature_names_in_)
        if len(cols) != len(vec):
            return None, (
                f"Model columns={len(cols)} but detector features={len(vec)}. "
                "Use a matching model/backend pair."
            )
    else:
        cols = [f"f{i}" for i in range(len(vec))]

    return pd.DataFrame([vec], columns=cols), None


def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        return image
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / float(w)
    target_h = max(1, int(h * scale))
    return cv2.resize(image, (target_width, target_h), interpolation=cv2.INTER_AREA)


def _camera_url_candidates(camera_url: str) -> list[str]:
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

    unique = []
    seen = set()
    for u in candidates:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _open_url_camera(camera_url: str) -> tuple[cv2.VideoCapture, Optional[str]]:
    for candidate in _camera_url_candidates(camera_url):
        cap = cv2.VideoCapture(candidate)
        if not cap.isOpened():
            cap.release()
            continue

        # Validate stream by pulling a few frames before accepting.
        ok = False
        for _ in range(20):
            ret, _frame = cap.read()
            if ret:
                ok = True
                break

        if ok:
            return cap, candidate

        cap.release()

    return cv2.VideoCapture(), None


def _open_camera(
    camera_index: int,
    camera_url: Optional[str] = None,
) -> tuple[cv2.VideoCapture, Optional[int], Optional[str]]:
    if camera_url:
        cap, used_url = _open_url_camera(camera_url)
        if cap.isOpened() and used_url:
            return cap, None, used_url
        return cv2.VideoCapture(), None, None

    if camera_index >= 0:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            return cap, camera_index, None
        cap.release()
        return cv2.VideoCapture(), None, None

    # Auto-probe common camera indices on Linux/Raspberry Pi using V4L2.
    for idx in range(0, 11):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap, idx, None
        cap.release()

    return cv2.VideoCapture(), None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-time sign inference from webcam")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained model (default: best available model in models/)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="Webcam index (use -1 to auto-detect)",
    )
    parser.add_argument(
        "--camera-url",
        default=None,
        help="IP camera stream URL (for mobile webcam/ipcam)",
    )
    parser.add_argument("--smoothing-window", type=int, default=7, help="Prediction smoothing window")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.45,
        help="Minimum confidence to accept a prediction",
    )
    parser.add_argument(
        "--use-roi",
        action="store_true",
        help="Enable centered ROI-only detection",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=0.55,
        help="Relative size of centered ROI box (0.3-0.9 recommended)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mediapipe_tasks", "mediapipe", "contour"],
        default="auto",
        help="Hand feature extraction backend",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="Resize display windows to this max width (0 disables resize)",
    )
    args = parser.parse_args()

    if args.camera_url and args.camera_url.startswith("https://"):
        args.camera_url = "http://" + args.camera_url[len("https://") :]

    model_path = resolve_model_path(args.model)
    clf, expected_feature_count = load_model(model_path)

    hands = build_hands_detector(preferred_backend=args.backend)
    print(f"Using backend: {hands.get('backend')}")
    cap, opened_index, opened_url = _open_camera(args.camera_index, args.camera_url)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera. If using mobile IP cam, verify URL and phone/Pi are on "
            "the same network. For Pi camera, verify detection with 'rpicam-hello --list-cameras'. "
            "Then retry with --camera-url <stream> or --camera-index -1."
        )
    if opened_url:
        print(f"Using camera URL: {opened_url}")
    else:
        print(f"Using camera index: {opened_index}")

    history = collections.deque(maxlen=max(1, args.smoothing_window))
    fail_count = 0
    last_reconnect = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                time.sleep(0.1)
                if fail_count >= 30 and (time.time() - last_reconnect) > 2.0:
                    cap.release()
                    cap, opened_index, opened_url = _open_camera(
                        args.camera_index,
                        args.camera_url,
                    )
                    last_reconnect = time.time()
                    fail_count = 0
                continue
            fail_count = 0

            h, w = frame.shape[:2]
            if args.use_roi:
                roi_scale = max(0.3, min(0.9, args.roi_scale))
                roi_w = int(w * roi_scale)
                roi_h = int(h * roi_scale)
                x1 = (w - roi_w) // 2
                y1 = (h - roi_h) // 2
                x2 = x1 + roi_w
                y2 = y1 + roi_h

                roi = frame[y1:y2, x1:x2]
                features, roi_annotated = extract_hand_features(roi, hands)
                dip_mask = preprocess_frame(roi)

                annotated = frame.copy()
                annotated[y1:y2, x1:x2] = roi_annotated
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 255, 120), 2)
            else:
                features, annotated = extract_hand_features(frame, hands)
                dip_mask = preprocess_frame(frame)
                roi_w = w
                roi_h = h

            lines = [
                "Mode: LIVE INFERENCE",
                "Keys: [q]=quit",
                f"Model: {model_path.name}",
                f"Backend: {hands.get('backend')}  MinConf: {args.min_confidence:.2f}",
            ]
            if features is None:
                history.clear()
                if args.use_roi:
                    lines.append("Hand: not detected (place hand inside center box)")
                else:
                    lines.append("Hand: not detected")
                view = put_status(annotated, lines, color=(0, 0, 255))
            else:
                bx1, by1, bx2, by2 = features.bbox
                box_area = max(1, (bx2 - bx1) * (by2 - by1))
                roi_area = max(1, roi_w * roi_h)
                area_ratio = box_area / roi_area

                if area_ratio < 0.02 or area_ratio > 0.65:
                    history.clear()
                    lines.extend(
                        [
                            "Hand size invalid for ROI",
                            f"Area ratio: {area_ratio:.2f} (adjust distance)",
                        ]
                    )
                    view = put_status(annotated, lines, color=(0, 0, 255))
                else:
                    x_row, frame_err = _build_inference_frame(
                        clf,
                        features.features,
                        expected_feature_count,
                    )
                    if frame_err:
                        history.clear()
                        lines.extend(
                            [
                                "Prediction disabled",
                                frame_err,
                                "Try another model (tasks_v2 / tasks / legacy).",
                            ]
                        )
                        view = put_status(annotated, lines, color=(0, 0, 255))
                    else:
                        try:
                            probs = clf.predict_proba(x_row)[0]
                            classes = clf.classes_
                            best_idx = int(probs.argmax())
                            raw_label = str(classes[best_idx])
                            conf = float(probs[best_idx])

                            top_idx = probs.argsort()[-3:][::-1]
                            top3 = ", ".join([f"{classes[i]}:{probs[i]:.2f}" for i in top_idx])

                            if conf >= args.min_confidence:
                                history.append(raw_label)
                                try:
                                    smooth_label = mode(history) if history else "UNSURE"
                                except StatisticsError:
                                    smooth_label = raw_label
                            else:
                                history.clear()
                                smooth_label = "UNSURE"

                            lines.extend(
                                [
                                    f"Prediction: {smooth_label}",
                                    f"Raw: {raw_label} ({conf:.2f})",
                                    f"ROI area ratio: {area_ratio:.2f}",
                                    f"Top3: {top3}",
                                ]
                            )

                            if conf < args.min_confidence:
                                lines.append("Low confidence - hold gesture steady")
                                view = put_status(annotated, lines, color=(0, 180, 255))
                            else:
                                view = put_status(annotated, lines)
                        except Exception as ex:
                            history.clear()
                            lines.extend(
                                [
                                    "Prediction error",
                                    str(ex),
                                ]
                            )
                            view = put_status(annotated, lines, color=(0, 0, 255))

            view_display = _resize_to_width(view, args.display_width)
            mask_display = _resize_to_width(dip_mask, args.display_width)

            cv2.imshow("Sign Recognition", view_display)
            cv2.imshow("DIP Mask", mask_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        close_hands_detector(hands)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
