import os, json, argparse, numpy as np, matplotlib.pyplot as plt, joblib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from model import HeartDiseaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GLOBAL_DIR  = os.path.join(ROOT, "data", "global")
ART         = os.path.join(ROOT, "artifacts")
PLOTS       = os.path.join(ART, "plots")
os.makedirs(ART, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)

WEIGHTS_FINAL = os.path.join(ART, "fedavg.weights.h5")
WEIGHTS_BEST  = os.path.join(ART, "fedavg.best.h5")
SEL_THR_JSON  = os.path.join(ART, "selected_thresholds.json")
ACTIVE_THR    = os.path.join(ART, "active_threshold.json")
CALIBRATOR    = os.path.join(ART, "platt_calibrator.joblib")

def plot_cm(cm, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def main(metric):
    X_test = np.load(os.path.join(GLOBAL_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(GLOBAL_DIR, "y_test.npy"))

    use_path = WEIGHTS_BEST if os.path.exists(WEIGHTS_BEST) else WEIGHTS_FINAL
    model = HeartDiseaseModel(input_dim=X_test.shape[1])
    _ = model(tf.zeros((1, X_test.shape[1]), dtype=tf.float32), training=False)
    model.load_weights(use_path)
    print(f"Loaded weights (H5) from {use_path}")

    # threshold: prefer ACTIVE (set during training based on headline metric).
    if os.path.exists(ACTIVE_THR):
        with open(ACTIVE_THR, "r") as f: active = json.load(f)
        thr = float(active["threshold"])
        metric = active.get("metric", metric).lower()
    else:
        with open(SEL_THR_JSON, "r") as f: sel = json.load(f)
        thr = float(sel[metric]["threshold"])

    # --- Calibrated probabilities (if calibrator present) ---
    raw = model.predict(X_test, verbose=0).ravel()
    if os.path.exists(CALIBRATOR):
        calibrator = joblib.load(CALIBRATOR)
        probs = calibrator.predict_proba(raw.reshape(-1,1))[:,1]
        print("Using calibrated probabilities.")
    else:
        probs = raw
        print("Calibrator not found; using raw probabilities.")

    y_pred = (probs >= thr).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, probs)
    ap   = average_precision_score(y_test, probs)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n=== FEDERATED EVALUATION (GLOBAL TEST) — {metric.upper()}-opt ===")
    print(f"Threshold: {thr:.3f}")
    print(f"Acc: {acc*100:.2f}% | BAcc: {bacc*100:.2f}% | AUC: {auc:.4f} | AP: {ap:.4f} | "
          f"P: {prec:.4f} R: {rec:.4f} F1: {f1:.4f}")

    # save plots/artifacts
    cm_path = os.path.join(PLOTS, f"federated_confusion_{metric}.png")
    roc_path = os.path.join(PLOTS, f"federated_roc_{metric}.png")
    pr_path  = os.path.join(PLOTS, f"federated_pr_{metric}.png")

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(5,4)); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend(); plt.tight_layout(); plt.savefig(roc_path, dpi=150); plt.close()

    precs, recs, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(5,4)); plt.plot(recs, precs, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall"); plt.legend(); plt.tight_layout(); plt.savefig(pr_path, dpi=150); plt.close()

    plot_cm(cm, cm_path, title=f"Confusion Matrix ({metric.upper()}-opt)")

    rep = classification_report(y_test, y_pred, digits=4, zero_division=0)
    with open(os.path.join(ART, f"classification_report_federated_{metric}.txt"), "w") as f:
        f.write(rep)
    out = {
        "metric": metric, "threshold": thr,
        "metrics": {"acc": acc, "bacc": bacc, "auc": float(auc), "ap": float(ap),
                    "precision": float(prec), "recall": float(rec), "f1": float(f1), "cm": cm.tolist()},
        "paths": {"roc": roc_path, "pr": pr_path, "cm": cm_path}
    }
    with open(os.path.join(ART, f"metrics_federated_{metric}.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(ACTIVE_THR, "w") as f:
        json.dump({"metric": metric, "threshold": float(thr)}, f, indent=2)

    print("\nSaved:")
    print(f"  JSON  → {os.path.join(ART, f'metrics_federated_{metric}.json')}")
    print(f"  Report→ {os.path.join(ART, f'classification_report_federated_{metric}.txt')}")
    print(f"  ROC   → {roc_path}")
    print(f"  PR    → {pr_path}")
    print(f"  CM    → {cm_path}")
    print(f"  Active threshold for GUI → {ACTIVE_THR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="f1", choices=["acc","f1","bacc"])
    args = ap.parse_args()
    main(args.metric)
