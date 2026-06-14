"""
Real-time sign detection bridge for braille printing.
Captures live sign predictions and buffers them.
"""

import collections
import threading
import time
from pathlib import Path
from typing import Optional, List, Callable
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd

# Import from common.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    build_hands_detector,
    close_hands_detector,
    extract_hand_features,
    preprocess_frame,
)

# Suppress noisy protobuf deprecation spam from mediapipe dependencies.
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)


class SignToBrailleCapture:
    """Capture signs from live camera and buffer detected text."""

    def __init__(self, 
                 model_path: str = "models/sign_model_tasks_v6_combo.joblib",
                 camera_url: str = "http://192.168.0.105:8080/video",
                 min_confidence: float = 0.60,
                 smoothing_window: int = 11,
                 use_roi: bool = True,
                 roi_scale: float = 0.5,
                 backend: str = "mediapipe",
                 show_preview: bool = True,
                 display_width: int = 800):
        """
        Initialize sign capture.
        
        Args:
            model_path: Path to trained sign model
            camera_url: IP camera URL
            min_confidence: Minimum confidence threshold
            smoothing_window: Prediction smoothing window size
            use_roi: Enable ROI mode
            roi_scale: ROI scale factor
            backend: Hand detection backend
        """
        self.model_path = Path(model_path)
        self.camera_url = camera_url
        self.min_confidence = min_confidence
        self.smoothing_window = max(1, smoothing_window)
        self.use_roi = use_roi
        self.roi_scale = max(0.3, min(0.9, roi_scale))
        self.backend = backend
        self.show_preview = show_preview
        self.display_width = max(0, int(display_width))
        
        # State
        self.history = collections.deque(maxlen=self.smoothing_window)
        self.buffer: List[str] = []
        self.current_prediction = None
        self.confidence = 0.0
        self.is_running = False
        self.cap = None
        self.model = None
        self.expected_feature_count = None
        self.hands = None
        self.last_smoothed_label = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._read_failures = 0
        self._last_reconnect = 0.0
        self._use_snapshot = False
        
        # Threading
        self.thread = None
        self.on_sign_detected: Optional[Callable[[str, float], None]] = None
        
    def load_model(self) -> bool:
        """Load trained sign classifier."""
        try:
            if not self.model_path.exists():
                print(f"Model not found: {self.model_path}")
                return False
            
            data = joblib.load(self.model_path)
            if "model" not in data:
                print("Invalid model file format")
                return False
            
            self.model = data["model"]
            self.expected_feature_count = data.get("feature_count")
            if self.expected_feature_count is None and hasattr(self.model, "n_features_in_"):
                self.expected_feature_count = int(self.model.n_features_in_)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _camera_url_candidates(self, camera_url: str) -> list[str]:
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

    def _open_camera(self) -> bool:
        """Open IP camera with fallback URL candidates."""
        try:
            self._use_snapshot = False
            for candidate in self._camera_url_candidates(self.camera_url):
                if urlparse(candidate).path == "/shot.jpg":
                    self.camera_url = candidate
                    self._use_snapshot = True
                    return True

                cap = cv2.VideoCapture(candidate)
                if not cap.isOpened():
                    cap.release()
                    continue

                ok = False
                for _ in range(20):
                    ret, _ = cap.read()
                    if ret:
                        ok = True
                        break

                if ok:
                    self.cap = cap
                    self.camera_url = candidate
                    return True

                cap.release()

            return False
        except Exception as e:
            print(f"Camera error: {e}")
            return False

    def _reconnect_camera(self) -> bool:
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        return self._open_camera()

    def _read_snapshot(self) -> Optional[np.ndarray]:
        try:
            with urlopen(self.camera_url, timeout=5) as resp:
                data = resp.read()
            if not data:
                return None
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def start(self) -> bool:
        """Start sign detection thread."""
        if not self.load_model():
            return False

        if self.camera_url.startswith("https://"):
            self.camera_url = "http://" + self.camera_url[len("https://") :]
        
        self.hands = build_hands_detector(preferred_backend=self.backend)
        
        if not self._open_camera():
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        """Stop sign detection and close resources."""
        self.is_running = False

        # Release camera early to unblock any reads
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        if self.hands:
            close_hands_detector(self.hands)
            self.hands = None
        with self._frame_lock:
            self._latest_frame = None

    def _capture_loop(self) -> None:
        """Main capture loop (runs in thread)."""
        from statistics import mode, StatisticsError
        
        while self.is_running:
            try:
                if self._use_snapshot:
                    frame = self._read_snapshot()
                    ret = frame is not None
                else:
                    ret, frame = self.cap.read()

                if not ret:
                    self._read_failures += 1
                    time.sleep(0.1)
                    if self._read_failures >= 30 and (time.time() - self._last_reconnect) > 2.0:
                        if self._reconnect_camera():
                            self._last_reconnect = time.time()
                            self._read_failures = 0
                    continue
                self._read_failures = 0
                
                h, w = frame.shape[:2]
                
                # Extract features
                if self.use_roi:
                    roi_w = int(w * self.roi_scale)
                    roi_h = int(h * self.roi_scale)
                    x1 = (w - roi_w) // 2
                    y1 = (h - roi_h) // 2
                    x2 = x1 + roi_w
                    y2 = y1 + roi_h
                    roi = frame[y1:y2, x1:x2]
                    features, roi_annotated = extract_hand_features(roi, self.hands)
                    annotated = frame.copy()
                    annotated[y1:y2, x1:x2] = roi_annotated
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 255, 120), 2)
                else:
                    features, annotated = extract_hand_features(frame, self.hands)
                
                # Process features
                if features is None:
                    self.history.clear()
                    self.current_prediction = None
                    self.confidence = 0.0
                    if self.show_preview:
                        self._put_text(annotated, ["Hand: not detected"])
                        self._set_latest_frame(annotated)
                    continue
                
                # Build inference frame
                vec = np.asarray(features.features, dtype=np.float32).reshape(-1)
                
                if self.expected_feature_count and len(vec) != int(self.expected_feature_count):
                    self.history.clear()
                    self.current_prediction = None
                    self.confidence = 0.0
                    if self.show_preview:
                        self._put_text(
                            annotated,
                            [
                                "Prediction disabled",
                                f"Model expects {int(self.expected_feature_count)} features",
                                f"Detector produced {len(vec)}",
                            ],
                        )
                        self._set_latest_frame(annotated)
                    continue
                
                cols = [f"f{i}" for i in range(len(vec))]
                x_row = pd.DataFrame([vec], columns=cols)
                
                # Predict
                try:
                    probs = self.model.predict_proba(x_row)[0]
                    classes = self.model.classes_
                    best_idx = int(probs.argmax())
                    raw_label = str(classes[best_idx])
                    conf = float(probs[best_idx])
                    
                    if conf >= self.min_confidence:
                        self.history.append(raw_label)
                        try:
                            smooth_label = mode(self.history) if self.history else "UNSURE"
                        except StatisticsError:
                            smooth_label = raw_label
                    else:
                        self.history.clear()
                        smooth_label = None
                    
                    # Trigger callback on new prediction
                    if smooth_label:
                        self.last_smoothed_label = smooth_label
                        self.current_prediction = smooth_label
                        self.confidence = conf
                        
                        if self.on_sign_detected:
                            self.on_sign_detected(smooth_label, conf)
                    if self.show_preview:
                        lines = [
                            f"Prediction: {smooth_label or 'UNSURE'}",
                            f"Raw: {raw_label} ({conf:.2f})",
                        ]
                        self._put_text(annotated, lines)
                        self._set_latest_frame(annotated)
                
                except Exception:
                    self.history.clear()
                    self.current_prediction = None
                    self.confidence = 0.0
                    if self.show_preview:
                        self._put_text(annotated, ["Prediction error"])
                        self._set_latest_frame(annotated)
            
            except Exception as e:
                print(f"Capture loop error: {e}")
            
            time.sleep(0.01)

    def _set_latest_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _put_text(self, frame: np.ndarray, lines: List[str]) -> None:
        y = 25
        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 24

    def add_to_buffer(self, sign: str) -> None:
        """Add detected sign to buffer."""
        if sign and sign not in ['UNSURE', 'NOTHING']:
            self.buffer.append(sign)

    def get_buffer_text(self) -> str:
        """Get buffered text."""
        return ''.join(self.buffer)

    def clear_buffer(self) -> None:
        """Clear text buffer."""
        self.buffer.clear()

    def get_frame_info(self) -> dict:
        """Get current frame info for UI."""
        return {
            'current_sign': self.current_prediction,
            'confidence': self.confidence,
            'buffer': self.get_buffer_text(),
            'buffer_count': len(self.buffer),
        }


if __name__ == "__main__":
    # Simple test
    capture = SignToBrailleCapture()
    
    def on_sign(sign: str, conf: float):
        print(f"Detected: {sign} ({conf:.2f})")
    
    capture.on_sign_detected = on_sign
    
    if capture.start():
        print("Capture started. Press Ctrl+C to stop.")
        try:
            while True:
                info = capture.get_frame_info()
                if info['current_sign']:
                    print(f"Current: {info['current_sign']}, Buffer: {info['buffer']}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            capture.stop()
    else:
        print("Failed to start capture")
