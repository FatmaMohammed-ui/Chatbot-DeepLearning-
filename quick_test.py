import joblib
import json
import numpy as np
from tensorflow.keras.models import load_model

# Paths
VECT_PATH = "vectorizer.joblib"
LE_PATH = "label_encoder.joblib"
MODEL_PATH = "model.h5"
RESP_PATH = "intents_responses.json"

# Load artifacts
vect = joblib.load(VECT_PATH)
le = joblib.load(LE_PATH)
model = load_model(MODEL_PATH)
with open(RESP_PATH, "r", encoding="utf-8") as f:
    responses = json.load(f)

# Sample test sentences for each intent
test_sentences = {
    "account_help": ["I need help with my account", "أحتاج مساعدة في حسابي"],
    "business_hours": ["What are your business hours?", "ما هي ساعات العمل؟"],
    "cancellation": ["I want to cancel my order", "أريد إلغاء طلبي"],
    "order_status": ["Where is my order?", "أين طلبي؟"],
    "password_reset": ["I forgot my password", "نسيت كلمة المرور الخاصة بي"],
    "payment_update": ["I want to update my payment method", "أريد تحديث طريقة الدفع الخاصة بي"],
    "return_request": ["I want to return my item", "أريد إرجاع المنتج الخاص بي"],
    "service_info": ["Tell me about your services", "أخبرني عن خدماتكم"],
    "technical_support": ["I need technical support", "أحتاج دعم فني"]
}

print("Quick Test of Intent Classifier\n" + "-"*40)

for intent, sentences in test_sentences.items():
    for sentence in sentences:
        x = vect.transform([sentence]).toarray()
        preds = model.predict(x)
        pred_idx = np.argmax(preds)
        detected_intent = le.inverse_transform([pred_idx])[0]
        confidence = preds[0][pred_idx]
        
        # Choose response (English if sentence is in English, else Arabic)
        lang = "en" if all(ord(c) < 128 for c in sentence) else "ar"
        resp = responses.get(detected_intent, {}).get(lang, ["(No response)"])[0]
        
        print(f"Input: {sentence}")
        print(f"Detected intent: {detected_intent}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Response: {resp}")
        print("-"*40)
