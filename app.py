from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model & encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le_location = joblib.load("label_encoders/location_encoder.pkl")
le_property = joblib.load("label_encoders/propertytype_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            bhk = float(request.form["bhk"])
            bathroom = float(request.form["bathroom"])
            sqft = float(request.form["area"])
            location = request.form["location"]
            propertytype = request.form["propertytype"]

            # Safe Encoding
            location_encoded = le_location.transform([location])[0] if location in le_location.classes_ else 0
            property_encoded = le_property.transform([propertytype])[0] if propertytype in le_property.classes_ else 0

            # ✅ VERY IMPORTANT: MATCH EXACT TRAINING FEATURES
            pricepersqft = sqft / 100   # Dummy calculation to satisfy model feature

            input_df = pd.DataFrame([{
                "bhk": bhk,
                "bathroom": bathroom,
                "sqft": sqft,
                "pricepersqft": pricepersqft,
                "location": location_encoded,
                "propertytype": property_encoded
            }])

            input_scaled = scaler.transform(input_df)
            prediction = round(model.predict(input_scaled)[0], 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
