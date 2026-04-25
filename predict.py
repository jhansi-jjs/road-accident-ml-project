import joblib
import numpy as np

# ✅ correct model name
model = joblib.load("models/random_forest_model.pkl")

sample = np.array([[1, 0, 45, 2, 60]])  # adjust later

prediction = model.predict(sample)

print("Predicted Severity:", prediction)
