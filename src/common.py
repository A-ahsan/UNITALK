from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


LABELS: List[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "SPACE",
    "DELETE",
    "NOTHING",
]

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


@dataclass
class HandFeatures:
    landmarks_px: List[Tuple[int, int]]
    features: np.ndarray
    bbox: Tuple[int, int, int, int]


def _normalize_points(points_px: List[Tuple[int, int]]) -> Optional[np.ndarray]:
    wrist_x, wrist_y = points_px[0]
    coords = np.array([[x - wrist_x, y - wrist_y] for (x, y) in points_px], dtype=np.float32)
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale < 1e-6:
        return None
    return (coords / scale).reshape(-1)


def _canonicalize_handedness(
    points_px: List[Tuple[int, int]], handedness_label: Optional[str]
) -> List[Tuple[int, int]]:
    if not handedness_label:
        return points_px

    # Convert left-hand landmarks to a right-hand canonical view.
    if handedness_label.strip().lower() != "left":
        return points_px

    wrist_x = points_px[0][0]
    return [(2 * wrist_x - x, y) for (x, y) in points_px]


def _sample_points_on_contour(contour: np.ndarray, n: int = 21) -> List[Tuple[int, int]]:
    pts = contour[:, 0, :]
    if len(pts) == 0:
        return []
    idxs = np.linspace(0, len(pts) - 1, num=n, dtype=int)
    return [(int(pts[i][0]), int(pts[i][1])) for i in idxs]


def _ensure_tasks_model() -> Path:
    model_path = Path("models") / "hand_landmarker.task"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    urlretrieve(url, model_path)
    return model_path


def build_hands_detector(
    min_detection_confidence: float = 0.6,
    min_tracking_confidence: float = 0.6,
    max_num_hands: int = 1,
    preferred_backend: str = "auto",
):
    backend = preferred_backend.lower().strip()

    if backend not in {"auto", "mediapipe", "mediapipe_tasks", "contour"}:
        backend = "auto"

    if backend in {"auto", "mediapipe"} and mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        return {"backend": "mediapipe", "detector": hands}

    if backend in {"auto", "mediapipe_tasks"} and mp is not None and hasattr(mp, "tasks"):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = _ensure_tasks_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        return {"backend": "mediapipe_tasks", "detector": detector}

    return {"backend": "contour", "detector": None}


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresh


def extract_hand_features(frame: np.ndarray, hands) -> Tuple[Optional[HandFeatures], np.ndarray]:
    backend = hands.get("backend", "contour") if isinstance(hands, dict) else "contour"

    if backend == "mediapipe":
        detector = hands["detector"]
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        if not result.multi_hand_landmarks:
            return None, frame

        hand_landmarks = result.multi_hand_landmarks[0]
        handedness_label: Optional[str] = None
        if result.multi_handedness:
            handedness_label = result.multi_handedness[0].classification[0].label

        points_px: List[Tuple[int, int]] = []
        xs: List[int] = []
        ys: List[int] = []

        for lm in hand_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points_px.append((x, y))
            xs.append(x)
            ys.append(y)

        points_px = _canonicalize_handedness(points_px, handedness_label)

        x_min = max(0, min(xs) - 20)
        y_min = max(0, min(ys) - 20)
        x_max = min(w, max(xs) + 20)
        y_max = min(h, max(ys) + 20)

        normalized = _normalize_points(points_px)
        if normalized is None:
            return None, frame

        features = HandFeatures(
            landmarks_px=points_px,
            features=normalized,
            bbox=(x_min, y_min, x_max, y_max),
        )

        annotated = frame.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
        )
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)

        return features, annotated

    if backend == "mediapipe_tasks":
        detector = hands["detector"]
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if not result.hand_landmarks:
            return None, frame

        hand_landmarks = result.hand_landmarks[0]
        handedness_label: Optional[str] = None
        if result.handedness and result.handedness[0]:
            handedness_label = result.handedness[0][0].category_name

        points_px: List[Tuple[int, int]] = []
        xs: List[int] = []
        ys: List[int] = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points_px.append((x, y))
            xs.append(x)
            ys.append(y)

        points_px = _canonicalize_handedness(points_px, handedness_label)

        x_min = max(0, min(xs) - 20)
        y_min = max(0, min(ys) - 20)
        x_max = min(w, max(xs) + 20)
        y_max = min(h, max(ys) + 20)

        normalized = _normalize_points(points_px)
        if normalized is None:
            return None, frame

        features = HandFeatures(
            landmarks_px=points_px,
            features=normalized,
            bbox=(x_min, y_min, x_max, y_max),
        )

        annotated = frame.copy()
        for i, j in HAND_CONNECTIONS:
            xi, yi = points_px[i]
            xj, yj = points_px[j]
            cv2.line(annotated, (xi, yi), (xj, yj), (0, 220, 120), 2)
        for x, y in points_px:
            cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)

        return features, annotated

    # Fallback backend: contour-derived pseudo landmarks.
    mask = preprocess_frame(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, frame

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1500:
        return None, frame

    points_px = _sample_points_on_contour(contour, n=21)
    if len(points_px) != 21:
        return None, frame

    xs = [p[0] for p in points_px]
    ys = [p[1] for p in points_px]
    x_min = max(0, min(xs) - 20)
    y_min = max(0, min(ys) - 20)
    x_max = min(frame.shape[1], max(xs) + 20)
    y_max = min(frame.shape[0], max(ys) + 20)

    normalized = _normalize_points(points_px)
    if normalized is None:
        return None, frame

    features = HandFeatures(
        landmarks_px=points_px,
        features=normalized,
        bbox=(x_min, y_min, x_max, y_max),
    )

    annotated = frame.copy()
    cv2.drawContours(annotated, [contour], -1, (255, 100, 0), 2)
    for x, y in points_px:
        cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)
    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)

    return features, annotated


def close_hands_detector(hands) -> None:
    if isinstance(hands, dict):
        detector = hands.get("detector")
        if detector is not None and hasattr(detector, "close"):
            detector.close()


def put_status(
    frame: np.ndarray,
    lines: List[str],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = frame.copy()
    y = 30
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 30
    return out
