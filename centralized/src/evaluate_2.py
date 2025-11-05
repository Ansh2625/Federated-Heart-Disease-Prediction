import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, balanced_accuracy_score,
    confusion_matrix, roc_curve
)
from model_2 import SAINT

# ===== Paths =====
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

# ===== Load Test Data =====
X_test = np.load(os.path.join(PROC, "X_test.npy"))
y_test = np.load(os.path.join(PROC, "y_test.npy"))
print(f"Loaded Test Set: {X_test.shape}, Positives={y_test.sum()}")

# ===== Load Model =====
model = SAINT(input_dim=X_test.shape[1], depth=8, dim=256, heads=4, dropout=0.3, ffn_mult=4)
ckpt_path = os.path.join(ART, "saint_best")
model.load_weights(ckpt_path).expect_partial()
print(f"[Loaded weights] {ckpt_path}")

# ===== Evaluate =====
probs = model.predict(X_test, batch_size=256).ravel()
preds = (probs >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
auc = roc_auc_score(y_test, probs)
bacc = balanced_accuracy_score(y_test, preds)

results = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc_roc": auc,
    "balanced_accuracy": bacc
}

print("\n=== Test Results ===")
print(json.dumps(results, indent=2))

with open(os.path.join(ART, "saint_eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("[Saved] Evaluation metrics JSON")

# ===== Confusion Matrix =====
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(ART, "confusion_matrix.png"))
plt.close()
print("[Saved] Confusion Matrix")

# ===== ROC Curve =====
fpr, tpr, _ = roc_curve(y_test, probs)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ART, "roc_curve.png"))
plt.close()
print("[Saved] ROC Curve")

# ===== Bar Chart of Metrics =====
plt.figure(figsize=(7, 4))
labels = list(results.keys())
values = list(results.values())
sns.barplot(x=labels, y=values)
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.title("Evaluation Metrics")
plt.tight_layout()
plt.savefig(os.path.join(ART, "metrics_barplot.png"))
plt.close()
print("[Saved] Metrics Bar Chart")
