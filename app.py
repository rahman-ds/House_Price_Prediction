from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ ABSOLUTELY NO PICKLE HERE
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
location_encoder = joblib.load("label_encoders/location_encoder.pkl")
property_encoder = joblib.load("label_encoders/propertytype_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            bhk = float(request.form["bhk"])
            bathroom = float(request.form["bathroom"])
            area = float(request.form["area"])
            location = request.form["location"]
            property_type = request.form["propertytype"]

            location_encoded = location_encoder.transform([location])[0]
            property_encoded = property_encoder.transform([property_type])[0]

            # ✅ EXACT FEATURE COUNT = 5
            features = np.array([[bhk, property_encoded, location_encoded, area, 0]])

            features_scaled = scaler.transform(features)
            result = model.predict(features_scaled)[0]
            prediction = round(result, 2)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
