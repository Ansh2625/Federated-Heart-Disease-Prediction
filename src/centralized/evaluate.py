# src/centralized/evaluate.py
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve)

# paths helper
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(THIS_DIR), "common"))
from paths import Paths
P = Paths()

from model import HeartDiseaseModel

# load data
X_val  = np.load(os.path.join(P.DATA_PROCESSED, "X_val.npy"))
y_val  = np.load(os.path.join(P.DATA_PROCESSED, "y_val.npy"))
X_test = np.load(os.path.join(P.DATA_PROCESSED, "X_test.npy"))
y_test = np.load(os.path.join(P.DATA_PROCESSED, "y_test.npy"))

# model (warm build) + weights
input_dim = X_test.shape[1]
model = HeartDiseaseModel(input_dim=input_dim)
_ = model(np.zeros((1, input_dim), dtype=np.float32), training=False)
model.load_weights(os.path.join(P.ARTIFACTS_CENT, "best_model.weights.h5"))

# threshold
thr = 0.5
thr_json = os.path.join(P.ARTIFACTS_CENT, "selected_threshold.json")
if os.path.exists(thr_json):
    with open(thr_json, "r") as f:
        thr = float(json.load(f)["threshold"])
print(f"Using threshold: {thr:.3f}")

def evaluate(name, X, y, prefix):
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= thr).astype(int)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    fpr, tpr, _ = roc_curve(y, probs); roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, preds)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}% | AUC: {roc_auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y, preds, zero_division=0))

    # plots
    plots = P.PLOTS_CENT
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No CHD","CHD"], yticklabels=["No CHD","CHD"])
    plt.title(f"{name} Confusion Matrix (thr={thr:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(plots, f"{prefix}_cm.png"), dpi=150); plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}"); plt.plot([0,1],[0,1],'--',color='gray')
    plt.title(f"{name} ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots, f"{prefix}_roc.png"), dpi=150); plt.close()

    pr, re, _ = precision_recall_curve(y, probs)
    pr_auc = auc(re, pr)
    plt.figure(figsize=(5,4))
    plt.plot(re, pr, label=f"AUC={pr_auc:.2f}")
    plt.title(f"{name} Precision-Recall"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots, f"{prefix}_pr.png"), dpi=150); plt.close()

evaluate("Validation", X_val, y_val, "val")
evaluate("Test", X_test, y_test, "test")
print("\nSaved plots →", P.PLOTS_CENT)
