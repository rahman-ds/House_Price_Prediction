from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

location_encoder = joblib.load(
    os.path.join(BASE_DIR, "label_encoders", "location_encoder.pkl")
)

# 👇 NOTICE: propertytype_encoder.pkl (correct name)
property_encoder = joblib.load(
    os.path.join(BASE_DIR, "label_encoders", "propertytype_encoder.pkl")
)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            bhk = float(request.form["bhk"])
            property_type = request.form["propertytype"]
            location = request.form["location"]
            sqft = float(request.form["sqft"])
            pricepersqft = float(request.form["pricepersqft"])

            # Encode categorical values
            property_encoded = property_encoder.transform([property_type])[0]
            location_encoded = location_encoder.transform([location])[0]

            # ✅ Exact order used in training:
            # ['bhk', 'propertytype', 'location', 'sqft', 'pricepersqft']
            features = np.array([[bhk, property_encoded, location_encoded, sqft, pricepersqft]])

            features_scaled = scaler.transform(features)
            result = model.predict(features_scaled)[0]

            # Format nicely with commas & 2 decimals
            prediction = f"{result:,.2f}"

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
