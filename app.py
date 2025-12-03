from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
location_encoder = pickle.load(open(os.path.join(BASE_DIR, "label_encoders/location_encoder.pkl"), "rb"))
property_encoder = pickle.load(open(os.path.join(BASE_DIR, "label_encoders/property_encoder.pkl"), "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            bhk = float(request.form["bhk"])
            bathroom = float(request.form["bathroom"])
            area = float(request.form["area"])
            location = request.form["location"]
            property_type = request.form["propertytype"]

            location_encoded = location_encoder.transform([location])[0]
            property_encoded = property_encoder.transform([property_type])[0]

            features = np.array([[bhk, area, location_encoded, property_encoded, 0]])
            features_scaled = scaler.transform(features)

            result = model.predict(features_scaled)[0]
            prediction = round(result, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
