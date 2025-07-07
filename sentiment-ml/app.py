from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('models/sentiment_model.pkl')

@app.route('/')
def home():
    return "✅ Sentiment API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get("review")

    if not review:
        return jsonify({"error": "Missing 'review' field"}), 400

    proba = model.predict_proba([review])[0]
    label = "positive" if proba[1] > 0.5 else "negative"
    confidence = round(proba[1] if label == "positive" else proba[0], 2)

    return jsonify({
        "label": label,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
