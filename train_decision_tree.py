"""
train_decision_tree.py
Member 2 — Decision Tree Model
22AIE304 Machine Learning Project: Road Accident Severity Prediction

Trains a DecisionTreeClassifier with hyperparameter tuning via GridSearchCV.
Loads the processed dataset produced by Member 1 (outputs/cleaned_dataset.csv).
Falls back to reproducing Member 1's preprocessing from the raw CSV if needed.

Outputs:
  models/decision_tree_model.pkl
  results/decision_tree_confusion_matrix.png
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
PROCESSED_CSV = ROOT / "outputs" / "cleaned_dataset.csv"
RAW_CSV       = ROOT / "data" / "UK_Accident.csv"
MODEL_OUT     = ROOT / "models" / "decision_tree_model.pkl"
RESULTS_DIR   = ROOT / "results"
CM_OUT        = RESULTS_DIR / "decision_tree_confusion_matrix.png"

# ── Split settings — identical to Member 1 ────────────────────────────────────
TARGET       = "Accident_Severity"
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load feature matrix X and target y.

    Priority:
      1. Member 1's exported cleaned_dataset.csv  (outputs/cleaned_dataset.csv)
      2. Fallback: reproduce Member 1's preprocessing from the raw CSV.
    """
    if PROCESSED_CSV.exists():
        print(f"[INFO] Loading processed dataset: {PROCESSED_CSV}")
        df = pd.read_csv(PROCESSED_CSV)
        print(f"[INFO] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        return X, y

    # ── Fallback: raw CSV ─────────────────────────────────────────────────────
    if not RAW_CSV.exists():
        print("[ERROR] Processed dataset not found and raw CSV is also missing.")
        print(f"         Expected processed : {PROCESSED_CSV}")
        print(f"         Expected raw       : {RAW_CSV}")
        print("[ERROR] Either run Member 1's notebook first, or place UK_Accident.csv in data/")
        sys.exit(1)

    print(f"[INFO] Processed CSV not found. Loading raw data: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"[INFO] Raw data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Reproduce Member 1's preprocessing exactly
    df = df.drop(columns=[
        "Unnamed: 0", "Accident_Index",
        "Location_Easting_OSGR", "Location_Northing_OSGR",
        "Longitude", "Latitude", "LSOA_of_Accident_Location",
    ], errors="ignore")

    df = df.drop(columns=["Special_Conditions_at_Site", "Carriageway_Hazards"], errors="ignore")

    df = df.fillna("Unknown")

    df = df.drop(columns=[
        "Date", "Time",
        "Local_Authority_(District)", "Local_Authority_(Highway)",
        "1st_Road_Number", "2nd_Road_Number",
    ], errors="ignore")

    df = pd.get_dummies(df, drop_first=True)
    print(f"[INFO] After preprocessing: {df.shape[0]:,} rows x {df.shape[1]} columns")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    X, y = load_data()

    # 2. Train-test split — identical settings to Member 1
    print("\n[INFO] Splitting data (test_size=0.2, random_state=42) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    # 3. Hyperparameter search via GridSearchCV
    param_grid = {
        "max_depth":         [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    }
    n_fits = (
        len(param_grid["max_depth"])
        * len(param_grid["min_samples_split"])
        * len(param_grid["min_samples_leaf"])
        * 3  # cv=3
    )
    print(f"\n[INFO] Starting GridSearchCV — {n_fits} fits (cv=3, scoring=f1_weighted)")
    print("[INFO] This may take several minutes on a 1.5M-row dataset ...")

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=1,      # sequential to avoid Windows paging-file limit with loky workers
        verbose=1,
        refit=True,  # automatically refit best params on full training set
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n[INFO] GridSearchCV completed in {elapsed:.1f}s")
    print(f"[INFO] Best parameters  : {grid_search.best_params_}")
    print(f"[INFO] Best CV F1 score : {grid_search.best_score_:.4f}")

    # 4. Evaluate on held-out test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {acc:.4f}  ({acc * 100:.2f}%)")
    print("\n[RESULT] Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[1, 2, 3],
        target_names=["High(1)", "Medium(2)", "Low(3)"],
        zero_division=0,
    ))

    # 5. Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
    RESULTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["High(1)", "Medium(2)", "Low(3)"],
        yticklabels=["High(1)", "Medium(2)", "Low(3)"],
        ax=ax,
    )
    ax.set_title("Decision Tree — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(CM_OUT, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] Confusion matrix  -> {CM_OUT}")

    # 6. Save model
    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)
    print(f"[SAVED] Model             -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
