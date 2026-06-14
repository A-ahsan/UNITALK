# Unified Assistive Communication - Software Stack (Laptop Webcam)

This folder now contains the complete software pipeline to run sign recognition with your laptop webcam.

## What is implemented

- Real-time webcam capture
- DIP preprocessing (grayscale -> Gaussian blur -> Otsu threshold)
- Hand landmark extraction (MediaPipe, 21 points)
- Landmark dataset collection tool
- ASL image dataset to landmark CSV converter
- Model training tool (SVM classifier)
- Real-time inference tool with prediction smoothing

## Project structure

- `src/common.py` - shared utilities (DIP + landmark extraction)
- `src/collect_landmarks.py` - capture labeled training samples from webcam
- `src/build_landmark_csv_from_asl.py` - convert ASL image folders to landmark CSV
- `src/train_classifier.py` - train and save model from collected CSV
- `src/realtime_inference.py` - run live predictions from webcam
- `data/landmarks.csv` - generated dataset (after collection)
- `models/sign_model.joblib` - trained model file (after training)

## Environment

A virtual environment is already configured in this workspace:

- Python: 3.14
- Packages: numpy, opencv-python, mediapipe, scikit-learn, joblib, pandas

## Run steps

### Option A: Use public ASL dataset (faster start)

1. Convert ASL image folders to landmark CSV

```powershell
d:/My_Project/sg/.venv/Scripts/python.exe src/build_landmark_csv_from_asl.py --max-per-class 500
```

2. Train classifier from generated CSV

```powershell
d:/My_Project/sg/.venv/Scripts/python.exe src/train_classifier.py --input data/landmarks_from_asl.csv
```

3. Run live inference

```powershell
d:/My_Project/sg/.venv/Scripts/python.exe src/realtime_inference.py
```

### Option B: Collect your own webcam dataset

1. Collect data

```powershell
d:/My_Project/sg/.venv/Scripts/python.exe src/collect_landmarks.py
```

Controls:
- `n` / `p`: next/previous label
- `c`: capture one sample for current label
- `q`: quit

2. Train classifier

```powershell
d:/My_Project/sg/.venv/Scripts/python.exe src/train_classifier.py
```

3. Run live inference

```powershell
d:/My_Project/sg/.venv/Scripts/python.exe src/realtime_inference.py
```

## Data collection guidance

- Collect at least 80-150 samples per label for better accuracy.
- Keep background and lighting varied.
- Include slight hand rotation and distance variation.
- Start with a smaller label set first (e.g., 5-10 signs), then expand.

## Notes

- This implementation is ready for webcam-based testing now.
- Hardware output routing (ESP32/Braille servos) can be connected next by adding a serial output module.
