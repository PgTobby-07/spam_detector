from flask import Flask, request, jsonify, render_template
import pickle
import os
import logging
import re

app = Flask(__name__, template_folder="templates")

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)

# =========================
# Load model
# =========================
MODEL_PATH = "spam_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Train model first: run 'python train.py'")

with open(MODEL_PATH, "rb") as f:
    model, vectorizer = pickle.load(f)

# =========================
# Light text cleaning
# =========================
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        text = None

        if data and "text" in data:
            text = data["text"]
        elif "text" in request.form:
            text = request.form["text"]

        if not text or not text.strip():
            return jsonify({"error": "No text provided"}), 400

        text_clean = clean_text(text)

        X_input = vectorizer.transform([text_clean])
        prediction = model.predict(X_input)[0]
        confidence = model.predict_proba(X_input).max()

        return jsonify({
            "prediction": prediction.lower(),   # ham / spam
            "confidence": round(float(confidence), 3),
            "preview": text[:100]
        })

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": "Internal server error"}), 500

# =========================
# Run app
# =========================
if __name__ == "__main__":
    print("Server running at: http://localhost:5000")
    if __name__ == "__main__":
        app.run()
