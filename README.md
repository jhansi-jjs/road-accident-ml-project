# 🚗 Road Accident Severity Prediction (Machine Learning Project)

## 📌 Project Overview
This project predicts the **severity of road accidents** using Machine Learning techniques.  
It analyzes factors such as weather conditions, road type, number of vehicles, and casualties to determine accident severity.

---

## 🎯 Objective
- Predict accident severity (1 = High, 2 = Medium, 3 = Low)
- Build a baseline ML model using Logistic Regression
- Perform data preprocessing and feature engineering
- Visualize accident data patterns

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Joblib  

---

## 📂 Project Structure
accident_ml_project/
│
├── data/                          # Raw dataset (excluded from version control)
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis & model training
├── models/
│   └── logistic_regression_model.pkl   # Trained ML model
├── outputs/
│   └── cleaned_dataset.csv       # Processed dataset (ignored)
├── src/                          # Source code (modular scripts - optional)
├── .gitignore                    # Files and folders excluded from Git
└── README.md                     # Project documentation

---

## 📊 Dataset
- Dataset: UK Road Accident Dataset  
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/silicon99/dft-accident-data  

---

## ⚙️ Workflow
1. Data Loading  
2. Data Cleaning (handling missing values)  
3. Feature Encoding (One-Hot Encoding)  
4. Train-Test Split  
5. Model Training (Logistic Regression)  
6. Model Evaluation  

---

## 📈 Results
- Model: Logistic Regression  
- Accuracy: **85.11%**

---

## 📊 Visualizations
- Accident Severity Distribution  
- Speed Limit Distribution  
- Weather Conditions Analysis  

---

## 💾 Model Saving
```python
import joblib
joblib.dump(model, "models/logistic_regression_model.pkl")
---
## 🚀 How to Run
git clone https://github.com/jhansi-jjs/road-accident-ml-project.git
cd accident_ml_project
pip install -r requirements.txt
jupyter notebook

## 📌 Future Improvements
- Implement advanced models such as Random Forest, Support Vector Machine (SVM), and XGBoost  
- Perform hyperparameter tuning to improve model performance  
- Develop a web-based application using Flask or Streamlit for real-time predictions  
- Apply feature selection techniques to enhance model efficiency  

---

## 👥 Team Contribution
- Member 1: Data preprocessing, feature engineering, and Logistic Regression model  
- Member 2: Decision Tree and Random Forest model development  
- Member 3: SVM model, model evaluation, and comparison  
