from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.pkl"


NUMERIC_FEATURES = [
    "amount",
    "previous_transactions",
    "transaction_velocity",
    "is_new_device",
    "hour",
]

CATEGORICAL_FEATURES = [
    "country",
    "payment_type",
    "device_type",
]


def train_fraud_model():
    df = pd.read_csv(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["fraud"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    print("\nFraud Detection Model Evaluation")
    print("=" * 40)
    print(classification_report(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        },
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")

    return model, preprocessor
def explain_fraud_risk(transaction):
    reasons = []

    amount = float(transaction.get("amount", 0))
    velocity = int(transaction.get("transaction_velocity", 0))
    new_device = int(transaction.get("is_new_device", 0))
    hour = int(transaction.get("hour", 0))
    previous_transactions = int(
        transaction.get("previous_transactions", 0)
    )

    if velocity >= 7:
        reasons.append(
            f"Very high transaction velocity ({velocity} transactions)"
        )
    elif velocity >= 5:
        reasons.append(
            f"Elevated transaction velocity ({velocity} transactions)"
        )

    if new_device == 1:
        reasons.append("Transaction originated from a new device")

    if hour <= 4:
        reasons.append(
            f"Transaction occurred during an unusual hour ({hour:02d}:00)"
        )

    if amount >= 19.99:
        reasons.append(
            f"High transaction amount ({amount:.2f})"
        )

    if previous_transactions <= 2:
        reasons.append(
            "Very limited previous transaction history"
        )

    if not reasons:
        reasons.append(
            "No major suspicious transaction indicators detected"
        )

    return reasons

def predict_fraud(transaction):
    if not MODEL_PATH.exists():
        train_fraud_model()

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]

    df = pd.DataFrame([transaction])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    X_processed = preprocessor.transform(X)

    probability = model.predict_proba(X_processed)[0][1]

    if probability >= 0.95:
     risk_level = "CRITICAL"
    elif probability >= 0.80:
     risk_level = "HIGH"
    elif probability >= 0.30:
     risk_level = "MEDIUM"
    else:
     risk_level = "LOW"

    reasons = explain_fraud_risk(transaction)

    return {
    "fraud_probability": round(float(probability), 4),
    "risk_level": risk_level,
    "is_fraud": bool(probability >= 0.50),
    "reasons": reasons,
     }
def get_transaction(transaction_id):
    df = pd.read_csv(DATA_PATH)

    matches = df[
        df["transaction_id"].astype(str) == str(transaction_id)
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()     
def predict_risk_levels(transactions):
    if not MODEL_PATH.exists():
        train_fraud_model()

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]

    X = transactions[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    X_processed = preprocessor.transform(X)

    probabilities = model.predict_proba(X_processed)[:, 1]

    def get_risk_level(probability):
     if probability >= 0.95:
        return "CRITICAL"
     elif probability >= 0.80:
        return "HIGH"
     elif probability >= 0.30:
        return "MEDIUM"
     else:
        return "LOW"

    risk_levels = [
        get_risk_level(float(probability))
        for probability in probabilities
    ]

    result = transactions.copy()

    result["fraud_probability"] = probabilities.round(4)
    result["risk_level"] = risk_levels

    return result     

if __name__ == "__main__":
    train_fraud_model()