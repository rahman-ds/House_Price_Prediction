from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le_location = joblib.load("label_encoders/location_encoder.pkl")
le_property = joblib.load("label_encoders/propertytype_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            bhk = int(request.form["bhk"])
            bathroom = int(request.form["bathroom"])
            area = float(request.form["area"])
            location = request.form["location"]
            propertytype = request.form["propertytype"]

            # Safe encoding
            location_encoded = (
                le_location.transform([location])[0]
                if location in le_location.classes_
                else 0
            )
            property_encoded = (
                le_property.transform([propertytype])[0]
                if propertytype in le_property.classes_
                else 0
            )

            input_df = pd.DataFrame([{
                "bhk": bhk,
                "bathroom": bathroom,
                "area": area,
                "location": location_encoded,
                "propertytype": property_encoded
            }])

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
