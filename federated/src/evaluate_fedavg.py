# federated/src/evaluate_fedavg.py
import os, json, numpy as np, matplotlib.pyplot as plt
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, classification_report
)
import pandas as pd

from model import HeartDiseaseModel

# -------- Paths --------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLIENTS_DIR = os.path.join(ROOT, "data", "clients")
GLOBAL_DIR  = os.path.join(ROOT, "data", "global")
ART   = os.path.join(ROOT, "artifacts")
PLOTS = os.path.join(ART, "plots")
os.makedirs(ART, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

WEIGHTS_TF = os.path.join(ART, "fedavg.weights")     # preferred
WEIGHTS_H5 = os.path.join(ART, "fedavg.weights.h5")  # fallback
THRESH_PATH = os.path.join(ART, "selected_threshold.json")

# -------- Utils --------
def ensure_built(model: tf.keras.Model, input_dim: int):
    _ = model(tf.zeros((1, input_dim), dtype=tf.float32), training=False)

def choose_threshold_from_val(y_true, probs, mode="f1"):
    grid = np.linspace(0.05, 0.99, 191)
    best_t, best_score = 0.5, -1.0
    for t in grid:
        yhat = (probs >= t).astype(int)
        if mode == "f1":
            score = f1_score(y_true, yhat, zero_division=0)
        elif mode == "acc":
            score = accuracy_score(y_true, yhat)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0,1]).ravel()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            score = tpr - fpr
        if score > best_score:
            best_t, best_score = t, score
    return float(best_t)

def metrics_at_threshold(y_true, probs, thr):
    y_pred = (probs >= thr).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "bal_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try: out["roc_auc"] = roc_auc_score(y_true, probs)
    except Exception: out["roc_auc"] = float("nan")
    try: out["avg_precision"] = average_precision_score(y_true, probs)
    except Exception: out["avg_precision"] = float("nan")
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
    return out, y_pred

def plot_roc(y_true, probs, path_png):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Federated)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path_png, dpi=150); plt.close()

def plot_pr(y_true, probs, path_png):
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Federated)"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(path_png, dpi=150); plt.close()

def plot_confusion(cm, path_png, title="Confusion Matrix (Federated)"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(2):
        for j in range(2): ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(path_png, dpi=150); plt.close()

# -------- Load data --------
X_val  = np.load(os.path.join(GLOBAL_DIR, "X_val.npy"))
y_val  = np.load(os.path.join(GLOBAL_DIR, "y_val.npy"))
X_test = np.load(os.path.join(GLOBAL_DIR, "X_test.npy"))
y_test = np.load(os.path.join(GLOBAL_DIR, "y_test.npy"))
input_dim = X_test.shape[1]

# -------- Load model & weights --------
model = HeartDiseaseModel(input_dim=input_dim)
ensure_built(model, input_dim)

loaded = False
if os.path.exists(WEIGHTS_TF):
    try:
        model.load_weights(WEIGHTS_TF)
        loaded = True
        print(f"Loaded weights (TF format) from {WEIGHTS_TF}")
    except Exception as e:
        print(f"TF-format load failed: {e}")

if (not loaded) and os.path.exists(WEIGHTS_H5):
    try:
        model.load_weights(WEIGHTS_H5)
        loaded = True
        print(f"Loaded weights (H5) from {WEIGHTS_H5}")
    except Exception as e:
        raise RuntimeError(
            f"Failed loading both TF and H5 weights.\nTF path: {WEIGHTS_TF}\nH5 path: {WEIGHTS_H5}\nLast error: {e}"
        )

if not loaded:
    raise FileNotFoundError(f"No weights found. Expected one of: {WEIGHTS_TF} or {WEIGHTS_H5}")

# -------- Threshold --------
if os.path.exists(THRESH_PATH):
    with open(THRESH_PATH, "r") as f:
        thr = float(json.load(f)["threshold"])
    thr_source = "loaded_from_artifacts"
else:
    val_probs = model.predict(X_val, verbose=0).ravel()
    thr = choose_threshold_from_val(y_val, val_probs, mode="f1")
    thr_source = "computed_from_validation"

# -------- Evaluate on TEST --------
test_probs = model.predict(X_test, verbose=0).ravel()
test_metrics, test_pred = metrics_at_threshold(y_test, test_probs, thr)
cls_report = classification_report(y_test, test_pred, digits=4, zero_division=0)

# Save outputs
with open(os.path.join(ART, "metrics_federated.json"), "w") as f:
    json.dump({"threshold": thr, "threshold_source": thr_source, "test_metrics": test_metrics}, f, indent=2)
with open(os.path.join(ART, "classification_report_federated.txt"), "w") as f:
    f.write(cls_report)

# Plots
roc_path = os.path.join(PLOTS, "federated_roc.png")
pr_path  = os.path.join(PLOTS, "federated_pr.png")
cm_path  = os.path.join(PLOTS, "federated_confusion.png")
try: plot_roc(y_test, test_probs, roc_path)
except Exception: pass
try: plot_pr(y_test, test_probs, pr_path)
except Exception: pass
plot_confusion(test_metrics["confusion_matrix"], cm_path)

# Per-client evaluation
rows = []
client_ids = sorted([d for d in os.listdir(CLIENTS_DIR) if d.startswith("client_")])
for cid in client_ids:
    Xc = np.load(os.path.join(CLIENTS_DIR, cid, "X.npy"))
    yc = np.load(os.path.join(CLIENTS_DIR, cid, "y.npy"))
    pc = model.predict(Xc, verbose=0).ravel()
    mc, _ = metrics_at_threshold(yc, pc, thr)
    rows.append({
        "client": cid, "n": int(len(yc)),
        "pos": int((yc==1).sum()), "neg": int((yc==0).sum()),
        "accuracy": mc["accuracy"], "bal_accuracy": mc["bal_accuracy"],
        "precision": mc["precision"], "recall": mc["recall"], "f1": mc["f1"],
        "roc_auc": mc["roc_auc"], "avg_precision": mc["avg_precision"],
        "tn": mc["confusion_matrix"][0][0], "fp": mc["confusion_matrix"][0][1],
        "fn": mc["confusion_matrix"][1][0], "tp": mc["confusion_matrix"][1][1],
    })

if rows:
    pd.DataFrame(rows).sort_values("client").to_csv(os.path.join(ART, "per_client_metrics.csv"), index=False)

# Console summary
print("\n=== FEDERATED EVALUATION (GLOBAL TEST) ===")
print(f"Threshold used: {thr:.3f}  ({thr_source})")
print(f"Accuracy       : {test_metrics['accuracy']*100:.2f}%")
print(f"Balanced Acc   : {test_metrics['bal_accuracy']*100:.2f}%")
print(f"ROC AUC        : {test_metrics['roc_auc']:.4f}")
print(f"Avg Precision  : {test_metrics['avg_precision']:.4f}")
print(f"Precision      : {test_metrics['precision']:.4f}")
print(f"Recall         : {test_metrics['recall']:.4f}")
print(f"F1-score       : {test_metrics['f1']:.4f}")
print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
print(np.array(test_metrics["confusion_matrix"]))
print("\nSaved:")
print(f"  JSON  → {os.path.join(ART,'metrics_federated.json')}")
print(f"  Report→ {os.path.join(ART,'classification_report_federated.txt')}")
print(f"  ROC   → {roc_path}")
print(f"  PR    → {pr_path}")
print(f"  CM    → {cm_path}")
if rows:
    print(f"  Per-client CSV → {os.path.join(ART,'per_client_metrics.csv')}")
