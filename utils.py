# utils.py
"""
Small helpers for loading models/artifacts and preprocessing.
"""

import joblib
import json
from tensorflow.keras.models import load_model

def load_all():
    vect = joblib.load("vectorizer.joblib")
    le = joblib.load("label_encoder.joblib")
    model = load_model("model.h5")
    with open("intents_responses.json", "r", encoding="utf-8") as f:
        responses = json.load(f)
    return vect, le, model, responses
