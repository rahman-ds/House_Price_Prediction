import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import numpy as np
import os

# Load model and preprocessors
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
le_location = joblib.load("label_encoders/location_encoder.pkl")
le_property = joblib.load("label_encoders/propertytype_encoder.pkl")

# Default values
defaults = {
    "bhk": 2,
    "bathroom": 1,
    "area": 1000,
    "location": "Unknown",
    "propertytype": "Flat"
}

# Prediction function
def predict_price():
    try:
        bhk = int(entry_bhk.get()) if entry_bhk.get() else defaults["bhk"]
        bathroom = int(entry_bathroom.get()) if entry_bathroom.get() else defaults["bathroom"]
        area = float(entry_area.get()) if entry_area.get() else defaults["area"]
        location = entry_location.get() or defaults["location"]
        propertytype = entry_propertytype.get() or defaults["propertytype"]

        # Safe encoding
        location_encoded = le_location.transform([location])[0] if location in le_location.classes_ else 0
        property_encoded = le_property.transform([propertytype])[0] if propertytype in le_property.classes_ else 0

        input_df = pd.DataFrame([{
            "bhk": bhk,
            "bathroom": bathroom,
            "area": area,
            "location": location_encoded,
            "propertytype": property_encoded
        }])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        messagebox.showinfo("Predicted Price", f"Estimated House Price: ₹{round(prediction, 2)}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI layout
root = tk.Tk()
root.title("House Price Predictor")

tk.Label(root, text="BHK:").grid(row=0, column=0, sticky="e")
entry_bhk = tk.Entry(root)
entry_bhk.grid(row=0, column=1)

tk.Label(root, text="Bathrooms:").grid(row=1, column=0, sticky="e")
entry_bathroom = tk.Entry(root)
entry_bathroom.grid(row=1, column=1)

tk.Label(root, text="Area (sq ft):").grid(row=2, column=0, sticky="e")
entry_area = tk.Entry(root)
entry_area.grid(row=2, column=1)

tk.Label(root, text="Location:").grid(row=3, column=0, sticky="e")
entry_location = tk.Entry(root)
entry_location.grid(row=3, column=1)

tk.Label(root, text="Property Type:").grid(row=4, column=0, sticky="e")
entry_propertytype = tk.Entry(root)
entry_propertytype.grid(row=4, column=1)

tk.Button(root, text="Predict", command=predict_price).grid(row=5, columnspan=2, pady=10)

root.mainloop()
