from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app, origins="*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gold_price_model.pkl")
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("Model loaded successfully")
except Exception as e:
    model = None
    print(f"Model load error: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        features = np.array([[
            float(data["SPX"]), float(data["USO"]),
            float(data["SLV"]), float(data["EUR_USD"]),
            float(data["Year"]), float(data["Month"]),
            float(data["Day"]), float(data["Quarter"])
        ]])
        prediction = model.predict(features)[0]
        return jsonify({"prediction": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)