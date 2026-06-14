from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sign classifier from landmark CSV")
    parser.add_argument("--input", default="data/landmarks.csv", help="Path to landmark CSV")
    parser.add_argument("--model-out", default="models/sign_model.joblib", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column")
    if len(df) < 50:
        raise ValueError("Need at least 50 samples to train a useful model")

    y = df["label"]
    X = df.drop(columns=["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced")),
        ]
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, zero_division=0))

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_count": X.shape[1]}, model_out)
    print(f"Model saved to: {model_out}")


if __name__ == "__main__":
    main()
