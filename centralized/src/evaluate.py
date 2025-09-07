import os, json, numpy as np, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve)
from model import HeartDiseaseModel

# paths (FIXED)
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

model = HeartDiseaseModel(input_dim=X_test.shape[1])
_ = model(np.zeros((1, X_test.shape[1]), dtype=np.float32), training=False)
model.load_weights(os.path.join(ART, "best_model.weights.h5"))

thr = 0.5
pth = os.path.join(ART, "selected_threshold.json")
if os.path.exists(pth):
    with open(pth, "r") as f: thr = float(json.load(f)["threshold"])
print(f"Using threshold: {thr:.3f}")

def evaluate(name, X, y, prefix):
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= thr).astype(int)
    acc, prec, rec, f1 = (accuracy_score(y, preds),
                          precision_score(y, preds, zero_division=0),
                          recall_score(y, preds, zero_division=0),
                          f1_score(y, preds, zero_division=0))
    fpr, tpr, _ = roc_curve(y, probs); roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}% | AUC: {roc_auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y, preds, zero_division=0))
    plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No CHD","CHD"], yticklabels=["No CHD","CHD"])
    plt.title(f"{name} Confusion Matrix (thr={thr:.3f})"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{prefix}_cm.png"), dpi=150); plt.close()
    pr, re, _ = precision_recall_curve(y, probs); pr_auc = auc(re, pr)
    plt.figure(figsize=(5,4)); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}"); plt.plot([0,1],[0,1],'--',color='gray')
    plt.title(f"{name} ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{prefix}_roc.png"), dpi=150); plt.close()
    plt.figure(figsize=(5,4)); plt.plot(re, pr, label=f"AUC={pr_auc:.2f}")
    plt.title(f"{name} Precision–Recall"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{prefix}_pr.png"), dpi=150); plt.close()

evaluate("Validation", X_val, y_val, "val")
evaluate("Test", X_test, y_test, "test")
print("\nSaved plots →", PLOTS)
