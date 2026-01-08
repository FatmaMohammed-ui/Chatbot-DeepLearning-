# app.py
"""
Streamlit app for chatting with the trained intent classifier.
Usage:
streamlit run app.py
"""

import streamlit as st
import joblib
import json
import random
import pickle
import numpy as np

from tensorflow.keras.models import load_model

# ================== Paths ==================
VECT_PATH = "vectorizer.joblib"
LE_PATH = "label_encoder.joblib"
MODEL_PATH = "model.h5"
RESP_PATH = "intents_responses.json"
TEST_PICKLE = "test_data.pkl"
CM_PNG = "confusion_matrix.png"

CONFIDENCE_THRESHOLD = 0.6   # ⭐ confidence control

# ================== Load artifacts ==================
@st.cache_resource
def load_artifacts():
    vect = joblib.load(VECT_PATH)
    le = joblib.load(LE_PATH)
    model = load_model(MODEL_PATH)

    with open(RESP_PATH, "r", encoding="utf-8") as f:
        responses = json.load(f)

    return vect, le, model, responses

vect, le, model, responses = load_artifacts()

# ================== UI ==================
st.set_page_config(page_title="Chatbot (Intent Classifier)", layout="centered")

st.title("Chatbot (Intent Classification) — English / Arabic")
lang = st.radio("Choose language / اختر اللغة:", ("English", "العربية"))

st.markdown("Type a message and the model will predict the intent and show a response.")

user_input = st.text_input("You:", "")

col1, col2 = st.columns(2)

# ================== Chat Logic ==================
with col1:
    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please type a message.")
        else:
            # Vectorize input
            x = vect.transform([user_input]).toarray()

            # Predict
            preds = model.predict(x)
            confidence = float(np.max(preds))
            pred_idx = int(np.argmax(preds))

            # Decide intent
            if confidence < CONFIDENCE_THRESHOLD:
                intent = "fallback"
            else:
                intent = le.inverse_transform([pred_idx])[0]

            st.write("**Detected intent:**", intent)
            st.write("**Confidence:**", round(confidence, 2))

            # Choose response
            if intent in responses:
                if lang == "English":
                    resp = random.choice(
                        responses[intent].get("en", ["(no English response)"])
                    )
                else:
                    resp = random.choice(
                        responses[intent].get("ar", ["(لا يوجد رد عربي)"])
                    )
            else:
                resp = "I don't have a response for this intent."

            st.success(resp)

# ================== Evaluation ==================
with col2:
    if st.button("Show Evaluation"):
        try:
            with open(TEST_PICKLE, "rb") as f:
                X_test, y_test, texts_test = pickle.load(f)

            st.write("Test examples:", len(y_test))
            st.image(CM_PNG, caption="Confusion Matrix")

        except Exception as e:
            st.error("Evaluation artifacts not found. Run evaluate.py first.")
            st.write(str(e))

# ================== Notes ==================
st.markdown("---")
# st.markdown(
    # """
# **Notes**
# - You can freely edit `intents_responses.json` without retraining.
# - Training files and model artifacts should remain unchanged.
# - `fallback` intent is handled manually (no training needed).
# """
# )



# streamlit run app.py
# venv\Scripts\activate
