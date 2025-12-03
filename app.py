from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
location_encoder = pickle.load(open("location_encoder.pkl", "rb"))
property_encoder = pickle.load(open("property_encoder.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Collect inputs (MATCH TRAINING COLUMNS)
            bhk = float(request.form["bhk"])
            property_type = request.form["propertytype"]
            location = request.form["location"]
            sqft = float(request.form["sqft"])
            pricepersqft = float(request.form["pricepersqft"])

            # Encode categorical values
            property_encoded = property_encoder.transform([property_type])[0]
            location_encoded = location_encoder.transform([location])[0]

            # ✅ EXACT FEATURE ORDER AS TRAINING
            features = np.array([[ 
                bhk,
                property_encoded,
                location_encoded,
                sqft,
                pricepersqft
            ]])

            # Scale
            features_scaled = scaler.transform(features)

            # Predict TOTAL PRICE
            result = model.predict(features_scaled)[0]
            prediction = round(result, 2)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
