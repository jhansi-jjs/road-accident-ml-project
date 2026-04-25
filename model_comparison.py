import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("outputs/cleaned_dataset.csv")

X = data.drop("Accident_Severity", axis=1)
y = data["Accident_Severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ FIXED MODEL PATHS
lr = joblib.load("models/logistic_regression_model.pkl")
dt = joblib.load("models/decision_tree_model.pkl")
rf = joblib.load("models/random_forest_model.pkl")
svm = joblib.load("models/svm_model.pkl")

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "SVM": svm
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    })

df = pd.DataFrame(results)
print(df)

df.to_csv("results/model_comparison.csv", index=False)

print("Model comparison done")
