import os, json, numpy as np, tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, balanced_accuracy_score,
                             roc_curve, precision_recall_curve, auc, confusion_matrix)
import matplotlib.pyplot as plt
from model_2 import SAINT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GLOBAL_DIR = os.path.join(ROOT, "data", "global")
ART_DIR    = os.path.join(ROOT, "artifacts")
PLOT_DIR   = os.path.join(ART_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

X_test = np.load(os.path.join(GLOBAL_DIR, "X_test.npy"))
y_test = np.load(os.path.join(GLOBAL_DIR, "y_test.npy"))

# ---------- match training config (depth=4, dim=128) ----------
model = SAINT(input_dim=X_test.shape[1], depth=4, dim=128, heads=4, dropout=0.3, ffn_mult=2)
_ = model(tf.zeros((1, X_test.shape[1])))  # build
ckpt = os.path.join(ART_DIR, "fedavg_saint_best_2")
status = model.load_weights(ckpt); status.expect_partial()
print(f"[Loaded FedAvg weights] {ckpt}")

probs = model.predict(X_test, batch_size=256, verbose=0).ravel()
preds = (probs >= 0.5).astype(int)

acc  = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec  = recall_score(y_test, preds, zero_division=0)
f1   = f1_score(y_test, preds, zero_division=0)
auc_roc = roc_auc_score(y_test, probs)
bacc = balanced_accuracy_score(y_test, preds)

results = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "auc_roc": float(auc_roc),
    "balanced_acc": float(bacc),
}
print("\n=== FedAvg Test Results ===")
print(json.dumps(results, indent=2))
with open(os.path.join(ART_DIR, "fedavg_results_2.json"), "w") as f:
    json.dump(results, f, indent=2)
print("[Saved] artifacts/fedavg_results_2.json")

# ---- plots ----
# ROC
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (FedAvg SAINT)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fedavg_roc_2.png"), dpi=160); plt.close()

# PR
prec_c, rec_c, _ = precision_recall_curve(y_test, probs)
pr_auc = auc(rec_c, prec_c)
plt.figure(); plt.plot(rec_c, prec_c, label=f"AP={pr_auc:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (FedAvg SAINT)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fedavg_pr_2.png"), dpi=160); plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
plt.figure(); im = plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix")
plt.colorbar(im); ticks = np.arange(2); plt.xticks(ticks, ["0","1"]); plt.yticks(ticks, ["0","1"])
for (i,j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fedavg_cm_2.png"), dpi=160); plt.close()

print(f"[Saved plots] artifacts/plots/fedavg_roc_2.png, fedavg_pr_2.png, fedavg_cm_2.png")
