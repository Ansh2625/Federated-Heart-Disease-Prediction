# centralized_model/predict_and_metrics.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, precision_recall_curve
)
from model import HeartDiseaseModel

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")

# ---------- load data ----------
X_val  = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val  = np.load(os.path.join(DATA_DIR, "y_val.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# ---------- load model + weights (warm build BEFORE load) ----------
input_dim = X_test.shape[1]
model = HeartDiseaseModel(input_dim=input_dim)

# warm build via a dummy forward pass (required for subclassed models)
_ = model(np.zeros((1, input_dim), dtype=np.float32), training=False)

weights_path = os.path.join(DATA_DIR, "best_model.weights.h5")
model.load_weights(weights_path)

# ---------- load selected threshold ----------
thr_path = os.path.join(DATA_DIR, "selected_threshold.json")
best_thr = 0.5
if os.path.exists(thr_path):
    with open(thr_path, "r") as f:
        best_thr = float(json.load(f)["threshold"])
print(f"Using threshold: {best_thr:.3f}")

# ---------- helper to evaluate ----------
def evaluate_split(name, X, y, save_prefix, thr):
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= thr).astype(int)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"\n=== {name} (thr={thr:.3f}) ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"AUC      : {roc_auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Specificity: {spec:.4f}")
    print("\nClassification Report:\n", classification_report(y, preds, zero_division=0))

    # CM
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No CHD","CHD"], yticklabels=["No CHD","CHD"])
    plt.title(f"{name} Confusion Matrix (thr={thr:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"{save_prefix}_cm.png"), dpi=150)
    plt.close()

    # ROC
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"{save_prefix}_roc.png"), dpi=150)
    plt.close()

    # Precision-Recall
    pr, re, _ = precision_recall_curve(y, probs)
    pr_auc = auc(re, pr)
    plt.figure(figsize=(5,4))
    plt.plot(re, pr, label=f"AUC={pr_auc:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{name} Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"{save_prefix}_pr.png"), dpi=150)
    plt.close()

# ---------- evaluate with the saved threshold ----------
evaluate_split("Validation", X_val, y_val, "val", best_thr)
evaluate_split("Test",       X_test, y_test, "test", best_thr)

print("\nSaved plots to:", DATA_DIR)
