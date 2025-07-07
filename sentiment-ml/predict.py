import sys
import joblib

if len(sys.argv) != 2:
    print("Usage: python predict.py \"<review text>\"")
    sys.exit(1)

review = sys.argv[1]

# Load the trained model
model = joblib.load('models/sentiment_model.pkl')

# Predict
proba = model.predict_proba([review])[0]
label = "positive" if proba[1] > 0.5 else "negative"
confidence = proba[1] if label == "positive" else proba[0]

print(f"Prediction: {label} (confidence: {confidence:.2f})")
