import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

data = pd.read_csv("outputs/cleaned_dataset.csv")

X = data.drop("Accident_Severity", axis=1)
y = data["Accident_Severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

joblib.dump(model, "models/svm_model.pkl")

print("SVM model trained")
