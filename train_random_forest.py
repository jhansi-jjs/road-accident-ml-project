"""
train_random_forest.py
Member 2 — Random Forest Model (Primary / Best Model)
22AIE304 Machine Learning Project: Road Accident Severity Prediction

Trains a RandomForestClassifier using RandomizedSearchCV.
On a 1.5M-row dataset a full CV is expensive, so hyperparameter search
is done on a 20% stratified subsample of the training set for speed.
The best configuration is then refit on the FULL training set before
final evaluation, so accuracy figures are not penalised.

Outputs:
  models/random_forest_model.pkl
  results/random_forest_confusion_matrix.png
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
PROCESSED_CSV = ROOT / "outputs" / "cleaned_dataset.csv"
RAW_CSV       = ROOT / "data" / "UK_Accident.csv"
MODEL_OUT     = ROOT / "models" / "random_forest_model.pkl"
RESULTS_DIR   = ROOT / "results"
CM_OUT        = RESULTS_DIR / "random_forest_confusion_matrix.png"

# ── Split settings — identical to Member 1 ────────────────────────────────────
TARGET       = "Accident_Severity"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Fraction of training data used for RandomizedSearchCV.
# Full 1.2M rows × 20 iters × 3 folds × 100+ trees is impractical;
# 20% subsample (~240K rows) gives reliable rankings in reasonable time.
TUNE_SAMPLE_FRAC = 0.20


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

    # 3. Subsample training data for the tuning phase
    # Doing RandomizedSearchCV on the full 1.2M-row training set
    # (20 iters x 3 folds = 60 RF fits) would take hours.
    # A stratified 20% subsample gives reliable hyperparameter rankings
    # much faster. The winning config is refit on the full training set.
    n_tune = int(len(X_train) * TUNE_SAMPLE_FRAC)
    print(f"\n[INFO] Subsampling {TUNE_SAMPLE_FRAC*100:.0f}% of training data for tuning "
          f"({n_tune:,} rows) ...")

    rng = np.random.default_rng(RANDOM_STATE)
    tune_idx = rng.choice(len(X_train), size=n_tune, replace=False)

    if hasattr(X_train, "iloc"):
        X_tune = X_train.iloc[tune_idx]
        y_tune = y_train.iloc[tune_idx]
    else:
        X_tune = X_train[tune_idx]
        y_tune = y_train[tune_idx]

    # 4. Hyperparameter search
    param_dist = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", None],
        "class_weight":      [None, "balanced"],
    }

    print("[INFO] Starting RandomizedSearchCV (n_iter=20, cv=3, scoring=f1_weighted) ...")

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        # RF itself uses n_jobs=-1 (C-level threads, no subprocess forking — safe on Windows)
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1_weighted",
        cv=3,
        n_jobs=1,      # sequential CV folds to avoid Windows paging-file limit with loky workers
        verbose=1,
        random_state=RANDOM_STATE,
        refit=False,   # we will manually refit on the full training set below
    )

    t0 = time.time()
    search.fit(X_tune, y_tune)
    tune_elapsed = time.time() - t0

    best_params = search.best_params_
    print(f"\n[INFO] Tuning completed in {tune_elapsed:.1f}s")
    print(f"[INFO] Best parameters  : {best_params}")
    print(f"[INFO] Best CV F1 score : {search.best_score_:.4f}  (on {TUNE_SAMPLE_FRAC*100:.0f}% subsample)")

    # 5. Refit best model on FULL training data
    print(f"\n[INFO] Refitting best model on full training set ({len(X_train):,} rows) ...")
    best_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t1 = time.time()
    best_model.fit(X_train, y_train)
    refit_elapsed = time.time() - t1
    print(f"[INFO] Refit completed in {refit_elapsed:.1f}s")

    # 6. Evaluate on held-out test set
    y_pred = best_model.predict(X_test)

    acc          = accuracy_score(y_test, y_pred)
    f1_macro     = f1_score(y_test, y_pred, average="macro")
    f1_weighted  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*55}")
    print(f"  RANDOM FOREST — TEST SET RESULTS")
    print(f"{'='*55}")
    print(f"  Accuracy          : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Macro F1          : {f1_macro:.4f}")
    print(f"  Weighted F1       : {f1_weighted:.4f}")
    print(f"{'='*55}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[1, 2, 3],
        target_names=["High(1)", "Medium(2)", "Low(3)"],
        zero_division=0,
    ))

    # 7. Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
    RESULTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=["High(1)", "Medium(2)", "Low(3)"],
        yticklabels=["High(1)", "Medium(2)", "Low(3)"],
        ax=ax,
    )
    ax.set_title("Random Forest — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(CM_OUT, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] Confusion matrix  -> {CM_OUT}")

    # 8. Save model
    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)
    print(f"[SAVED] Model             -> {MODEL_OUT}")

    total = tune_elapsed + refit_elapsed
    print(f"\n[INFO] Total runtime: {total:.1f}s")


if __name__ == "__main__":
    main()
