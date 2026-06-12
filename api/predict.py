from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app, origins="*")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../backend/gold_price_model.pkl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    model = None

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200
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