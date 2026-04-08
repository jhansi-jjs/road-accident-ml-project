"""
feature_importance.py
Member 2 — Feature Importance Analysis
22AIE304 Machine Learning Project: Road Accident Severity Prediction

Loads the trained Random Forest model and plots the top feature importances.
Feature names are read directly from the model (sklearn >= 1.0 stores them
in feature_names_in_), so there is no risk of misalignment with encoded columns.

Outputs:
  results/feature_importance.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
MODEL_PATH  = ROOT / "models" / "random_forest_model.pkl"
RESULTS_DIR = ROOT / "results"
FI_OUT      = RESULTS_DIR / "feature_importance.png"

TOP_N = 15   # number of features to plot


def main():
    # 1. Load the trained Random Forest model
    if not MODEL_PATH.exists():
        print(f"[ERROR] Random Forest model not found: {MODEL_PATH}")
        print("[ERROR] Run train_random_forest.py first.")
        sys.exit(1)

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # 2. Retrieve feature names
    # sklearn >= 1.0 stores input feature names in feature_names_in_ when the
    # model is fit on a DataFrame — this guarantees alignment with the encoded
    # column names produced by pd.get_dummies in Member 1's preprocessing.
    if hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_
    else:
        # Fallback for older sklearn or non-DataFrame input
        feature_names = np.array([f"feature_{i}" for i in range(model.n_features_in_)])
        print("[WARN] feature_names_in_ not available. Using generic feature labels.")

    # 3. Extract importances and sort descending
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names  = feature_names[indices]
    sorted_scores = importances[indices]

    # 4. Print top 10
    print(f"\n[RESULT] Top 10 Feature Importances (Random Forest):")
    print(f"  {'Rank':<5} {'Feature':<52} {'Importance':>10}")
    print("  " + "-" * 69)
    for rank, (name, score) in enumerate(zip(sorted_names[:10], sorted_scores[:10]), 1):
        print(f"  {rank:<5} {str(name):<52} {score:>10.4f}")

    # 5. Bar chart of top TOP_N features
    top_names  = sorted_names[:TOP_N]
    top_scores = sorted_scores[:TOP_N]

    RESULTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reversed so the most important feature is at the top of the chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, TOP_N))
    bars = ax.barh(range(TOP_N), top_scores[::-1], color=colors)
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels([str(n) for n in top_names[::-1]], fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title(f"Random Forest — Top {TOP_N} Feature Importances\n"
                 f"(Accident Severity Prediction)")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)

    plt.tight_layout()
    fig.savefig(FI_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVED] Feature importance chart -> {FI_OUT}")


if __name__ == "__main__":
    main()
