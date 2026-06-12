from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gold_price_model.pkl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ Model loaded successfully")
except FileNotFoundError:
    model = None
    print("⚠️  Model file not found. Place gold_price_model.pkl in the backend folder.")

FEATURE_ORDER = ["SPX", "USO", "SLV", "EUR/USD", "Year", "Month", "Day", "Quarter"]


def _get(data, *keys):
    """Return first matching key from data."""
    for k in keys:
        if k in data:
            return data[k]
    raise KeyError(keys[0])


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Gold Price Prediction API is running!",
        "status": "ok",
        "features": FEATURE_ORDER,
        "num_features": len(FEATURE_ORDER),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please add gold_price_model.pkl to the backend folder."}), 500

    try:
        data = request.get_json()

        spx     = float(_get(data, "SPX"))
        uso     = float(_get(data, "USO"))
        slv     = float(_get(data, "SLV"))
        eur_usd = float(_get(data, "EUR_USD", "EUR/USD"))
        year    = int(float(_get(data, "Year")))
        month   = int(float(_get(data, "Month")))
        day     = int(float(_get(data, "Day")))

        # Quarter is optional — derive from month if not provided
        if "Quarter" in data and data["Quarter"] not in (None, ""):
            quarter = int(float(data["Quarter"]))
        else:
            quarter = (month - 1) // 3 + 1

        # Basic validation
        if not (1 <= month <= 12):
            return jsonify({"error": "Month must be between 1 and 12"}), 400
        if not (1 <= day <= 31):
            return jsonify({"error": "Day must be between 1 and 31"}), 400
        if not (1 <= quarter <= 4):
            return jsonify({"error": "Quarter must be between 1 and 4"}), 400

        features = np.array([[spx, uso, slv, eur_usd, year, month, day, quarter]])
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": round(float(prediction), 2),
            "inputs": {
                "SPX": spx,
                "USO": uso,
                "SLV": slv,
                "EUR/USD": eur_usd,
                "Year": year,
                "Month": month,
                "Day": day,
                "Quarter": quarter,
            },
            "unit": "USD per oz",
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "num_features": len(FEATURE_ORDER),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
