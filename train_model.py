"""Train the heart disease classifier and save it with joblib.

Run from the project root:
    pip install -r requirements-train.txt
    python train_model.py

Outputs:
    model/heart_model.joblib   - the trained scikit-learn model (loaded by the API)
    model/model_info.json      - training metadata shown by the /info endpoint
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_PATH = ROOT / "model" / "heart_model.joblib"
INFO_PATH = ROOT / "model" / "model_info.json"

# The order of this list is the order the API uses for prediction (keep it in sync with app/schemas.py)
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"   # 1 = heart disease present, 0 = absent


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # This Kaggle file contains many duplicated patients. Duplicates would leak between the
    # train and test sets and inflate the accuracy, so they are removed first.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"{len(df)} unique rows after dropping duplicates")

    # numpy arrays (not a DataFrame) are used so the API can predict from a plain list of numbers
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["no disease", "disease"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    info = {
        "algorithm": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 200, "max_depth": 6, "random_state": 42},
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_accuracy": round(float(accuracy), 4),
        "scikit_learn_version": sklearn.__version__,
        "dataset": "Heart Disease Dataset (Kaggle, johnsmith88) - duplicates removed",
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"Saved model to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
