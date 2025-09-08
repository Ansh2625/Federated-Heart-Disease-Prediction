import os, json, joblib, numpy as np, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, auc,
                             precision_recall_curve, balanced_accuracy_score)
from model import HeartDiseaseModel

# paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
ART  = os.path.join(ROOT, "artifacts")
PLOTS = os.path.join(ART, "plots")
os.makedirs(PLOTS, exist_ok=True)

X_val  = np.load(os.path.join(PROC, "X_val.npy"))
y_val  = np.load(os.path.join(PROC, "y_val.npy"))
X_test = np.load(os.path.join(PROC, "X_test.npy"))
y_test = np.load(os.path.join(PROC, "y_test.npy"))

# load model
model = HeartDiseaseModel(input_dim=X_test.shape[1])
_ = model(np.zeros((1, X_test.shape[1]), dtype=np.float32), training=False)
model.load_weights(os.path.join(ART, "best_model.weights.h5"))

# load threshold + calibrator
thr = 0.5
pth = os.path.join(ART, "selected_threshold.json")
if os.path.exists(pth):
    with open(pth, "r") as f:
        thr = float(json.load(f)["threshold"])
calib_path = os.path.join(ART, "platt_calibrator.joblib")
calibrator = joblib.load(calib_path) if os.path.exists(calib_path) else None
print(f"Using threshold: {thr:.3f}; Calibrator: {'yes' if calibrator else 'no'}")

def calibrated_probs(raw):
    if calibrator is None:
        return raw
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

def evaluate(name, X, y, prefix):
    raw = model.predict(X, verbose=0).ravel()
    probs = calibrated_probs(raw)
    preds = (probs >= thr).astype(int)

    acc  = accuracy_score(y, preds)
    bacc = balanced_accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    fpr, tpr, _ = roc_curve(y, probs); roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, preds)

    print(f"\n=== {name} ===")
    print(f"Acc: {acc*100:.2f}% | BalAcc: {bacc*100:.2f}% | AUC: {roc_auc:.4f} | "
          f"Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y, preds, zero_division=0))

    # CM
    plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No CHD","CHD"], yticklabels=["No CHD","CHD"])
    plt.title(f"{name} Confusion Matrix (thr={thr:.3f})"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{prefix}_cm.png"), dpi=150); plt.close()

    # ROC
    plt.figure(figsize=(5,4)); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{name} ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{prefix}_roc.png"), dpi=150); plt.close()

    # PR curve
    pr, re, _ = precision_recall_curve(y, probs); pr_auc = auc(re, pr)
    plt.figure(figsize=(5,4)); plt.plot(re, pr, label=f"AUC={pr_auc:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name} Precision–Recall")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{prefix}_pr.png"), dpi=150); plt.close()

evaluate("Validation", X_val, y_val, "val")
evaluate("Test", X_test, y_test, "test")
print("\nSaved plots →", PLOTS)
