# centralized/src/evaluate_2.py
import os, json, numpy as np, tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from model_2 import SAINT

# ===== Paths =====
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
ART  = os.path.join(ROOT, "artifacts")

# ===== Load Data =====
X_test = np.load(os.path.join(PROC, "X_test.npy"))
y_test = np.load(os.path.join(PROC, "y_test.npy"))
print(f"Loaded Test Set: {X_test.shape}, Positives={y_test.sum()}")

# ===== Model =====
model = SAINT(input_dim=X_test.shape[1], depth=8, dim=256, heads=4, dropout=0.3, ffn_mult=4)
ckpt_path = os.path.join(ART, "saint_best")
model.load_weights(ckpt_path).expect_partial() 
print(f"[Loaded weights] {ckpt_path}")

# ===== Evaluation =====
probs = model.predict(X_test, batch_size=256).ravel()
preds = (probs >= 0.5).astype(int)

acc  = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec  = recall_score(y_test, preds, zero_division=0)
f1   = f1_score(y_test, preds, zero_division=0)
auc  = roc_auc_score(y_test, probs)
bacc = balanced_accuracy_score(y_test, preds)

results = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc_roc": auc,
    "balanced_acc": bacc,
}

print("\n=== Test Results ===")
print(json.dumps(results, indent=2))

with open(os.path.join(ART, "saint_eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"[Saved] artifacts/saint_eval_results.json")
