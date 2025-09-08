import os, json, joblib, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from model import HeartDiseaseModel

# paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
ART  = os.path.join(ROOT, "artifacts")
PLOTS = os.path.join(ART, "plots")
os.makedirs(ART, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)

# load
X_train = np.load(os.path.join(PROC, "X_train.npy"))
X_val   = np.load(os.path.join(PROC, "X_val.npy"))
X_test  = np.load(os.path.join(PROC, "X_test.npy"))
y_train = np.load(os.path.join(PROC, "y_train.npy"))
y_val   = np.load(os.path.join(PROC, "y_val.npy"))
y_test  = np.load(os.path.join(PROC, "y_test.npy"))

print("Class distribution (train):", np.bincount(y_train))

# ===== 1) Class-aware training =====
classes = np.unique(y_train)
cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
print("Class weights:", class_weight)

model = HeartDiseaseModel(input_dim=X_train.shape[1])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
        tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc_roc", patience=12, mode="max",
                                     restore_best_weights=True, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200, batch_size=64,
    class_weight=class_weight,
    callbacks=callbacks, verbose=2
)

# ===== 2) Probability calibration (Platt scaling) on validation =====
val_raw = model.predict(X_val, verbose=0).ravel()
calibrator = LogisticRegression(solver="liblinear")
calibrator.fit(val_raw.reshape(-1, 1), y_val)

# Save calibrator to artifacts
joblib.dump(calibrator, os.path.join(ART, "platt_calibrator.joblib"))

# ===== 3) Threshold selection on validation (maximize F1, tie-break by balanced accuracy) =====
val_probs = calibrator.predict_proba(val_raw.reshape(-1, 1))[:, 1]

def metrics_at_threshold(y_true, p, t):
    pred = (p >= t).astype(int)
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    bacc = balanced_accuracy_score(y_true, pred)
    return acc, prec, rec, f1, bacc

best = {"t": 0.5, "acc": -1, "prec": -1, "rec": -1, "f1": -1, "bacc": -1}
for t in np.linspace(0.05, 0.95, 181):
    acc, prec, rec, f1, bacc = metrics_at_threshold(y_val, val_probs, t)
    # Primary: F1; Secondary: balanced accuracy; Tertiary: recall
    better = (f1 > best["f1"]) or (f1 == best["f1"] and bacc > best["bacc"]) or \
             (f1 == best["f1"] and bacc == best["bacc"] and rec > best["rec"])
    if better:
        best = {"t": float(t), "acc": acc, "prec": prec, "rec": rec, "f1": f1, "bacc": bacc}

best_thr = best["t"]
with open(os.path.join(ART, "selected_threshold.json"), "w") as f:
    json.dump({"threshold": best_thr, "val_metrics": best}, f, indent=2)
print(f"Selected threshold (by F1): {best_thr:.3f}")
print("Val @thr:", best)

# ===== 4) Final test evaluation with calibrated probs =====
test_raw = model.predict(X_test, verbose=0).ravel()
test_probs = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
y_pred = (test_probs >= best_thr).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, test_probs)
bacc = balanced_accuracy_score(y_test, y_pred)

print("\n=== FINAL TEST METRICS (centralized) ===")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Balanced : {bacc*100:.2f}%")
print(f"AUC-ROC  : {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")

# ===== 5) Plots =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.xticks([0, 1], ["No CHD", "CHD"]); plt.yticks([0, 1], ["No CHD", "CHD"])
plt.title(f"Confusion Matrix (thr={best_thr:.3f})"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "test_cm.png"), dpi=150); plt.close()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.plot(history.history["auc_roc"]); plt.plot(history.history["val_auc_roc"]); plt.title("AUC-ROC"); plt.legend(["train","val"])
plt.subplot(1, 2, 2); plt.plot(history.history["auc_pr"]);  plt.plot(history.history["val_auc_pr"]);  plt.title("AUC-PR");  plt.legend(["train","val"])
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "training_curves_auc.png"), dpi=150); plt.close()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"]); plt.title("Loss"); plt.legend(["train","val"])
plt.subplot(1, 2, 2); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"]); plt.title("Accuracy"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "training_curves_loss_acc.png"), dpi=150); plt.close()

# Save final weights
model.save_weights(os.path.join(ART, "best_model.weights.h5"))
