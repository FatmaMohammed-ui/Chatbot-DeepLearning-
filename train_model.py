"""
Train intent classification model from chatbot_intent_classification.csv
Produces:
- model.h5
- vectorizer.joblib
- label_encoder.joblib
- intents_responses.json (if not exists, creates default responses)
- test_data.pkl (X_test, y_test, texts_test) for evaluation
"""

import os
import json
import pickle
import random

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# ===================== Paths =====================
DATA_CSV = "chatbot_intent_classification.csv"
VECT_PATH = "vectorizer.joblib"
LE_PATH = "label_encoder.joblib"
MODEL_PATH = "model.h5"
RESP_PATH = "intents_responses.json"
TEST_PICKLE = "test_data.pkl"

# ===================== Default Responses =====================
def create_default_responses(intents):
    """Generate simple default responses in English & Arabic for every intent."""
    base_en = {
        "greet": ["Hello! How can I help you?", "Hi there! What can I do for you?"],
        "bye": ["Goodbye!", "See you later!"],
        "thanks": ["You're welcome!", "Happy to help!"],
    }
    base_ar = {
        "greet": ["أهلاً! كيف أستطيع مساعدتك؟", "مرحبا! ماذا تريد؟"],
        "bye": ["مع السلامة!", "أراك لاحقًا!"],
        "thanks": ["عفواً!", "سعيد بمساعدتك!"],
    }

    responses = {}
    for it in intents:
        r = {}
        r["en"] = base_en[it] if it in base_en else [f"Intent detected: {it}. I don't have a specific English response yet."]
        r["ar"] = base_ar[it] if it in base_ar else [f"النية: {it}. لا يوجد رد عربي محدد بعد."]
        responses[it] = r
    return responses

# ===================== Load Dataset =====================
def load_dataset(csv_path=DATA_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    # Ensure the expected columns exist
    if not {"user_input", "intent"}.issubset(df.columns):
        raise ValueError("CSV must contain 'user_input' and 'intent' columns.")
    df = df.dropna(subset=["user_input", "intent"])
    df = shuffle(df, random_state=42).reset_index(drop=True)
    texts = df["user_input"].astype(str).tolist()
    labels = df["intent"].astype(str).tolist()
    return texts, labels

# ===================== Build & Train Model =====================
def build_and_train(texts, labels):
    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train/test split
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, texts, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = len(np.unique(y))
                       
    # Build Keras model
    model = Sequential(
        Dense(128, activation="relu", input_shape=(X.shape[1],)),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # Save artifacts
    model.save(MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)
    joblib.dump(le, LE_PATH)
          
    # Save test split for evaluation
    with open(TEST_PICKLE, "wb") as f:
        pickle.dump((X_test, y_test, texts_test), f)

    return model, vectorizer, le

# ===================== Ensure Responses File =====================
def ensure_responses_file(intents):
    if os.path.exists(RESP_PATH):
        print(f"{RESP_PATH} exists — will use it.")
        with open(RESP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # verify all intents present
        for it in intents:
            if it not in data:
                print(f"Intent '{it}' missing in {RESP_PATH}. Adding default.")
                data[it] = create_default_responses([it])[it]
        with open(RESP_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        print(f"{RESP_PATH} not found — creating default responses.")
        data = create_default_responses(intents)
        with open(RESP_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Responses saved to {RESP_PATH}")

# ===================== Main =====================
def main():




    print("Loading dataset...")
    texts, labels = load_dataset(DATA_CSV)
    intents = sorted(list(set(labels)))
    print(f"Found {len(texts)} examples, {len(intents)} intents.")

    print("Ensuring responses file...")
    ensure_responses_file(intents)

    print("Training model...")
    model, vectorizer, le = build_and_train(texts, labels)

    print("Training complete. Artifacts saved:")
    print(" -", MODEL_PATH)
    print(" -", VECT_PATH)
    print(" -", LE_PATH)
    print(" -", TEST_PICKLE)
    print(" -", RESP_PATH)

if __name__ == "__main__":
    main()
