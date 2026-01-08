# evaluate.py
"""
Load saved artifacts and evaluate on the held-out test split.
Outputs:
- prints accuracy & classification report
- saves confusion_matrix.png
"""

import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns  # optional but allowed

MODEL_PATH = "model.h5"
VECT_PATH = "vectorizer.joblib"
LE_PATH = "label_encoder.joblib"
TEST_PICKLE = "test_data.pkl"
CM_PNG = "confusion_matrix.png"

from tensorflow.keras.models import load_model

def load_test_data():
    with open(TEST_PICKLE, "rb") as f:
        X_test, y_test, texts_test = pickle.load(f)
    return X_test, y_test, texts_test

def main():
    print("Loading artifacts...")
    model = load_model(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    le = joblib.load(LE_PATH)

    X_test, y_test, texts_test = load_test_data()

    print("Predicting...")
    preds_proba = model.predict(X_test)
    preds = np.argmax(preds_proba, axis=1)

    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}\n")

    labels = le.inverse_transform(sorted(list(set(y_test))))
    # classification report using original label names
    target_names = le.inverse_transform(sorted(list(set(y_test))))
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10,8))
    # Use seaborn heatmap for nicer display, but do not set colors explicitly
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PNG)
    print(f"Confusion matrix saved to {CM_PNG}")

if __name__ == "__main__":
    main()
