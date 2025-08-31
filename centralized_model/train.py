# centralized_model/train.py
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from model import HeartDiseaseModel

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")

# ---------- load ----------
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Class distribution (train):", np.bincount(y_train))

# ---------- model ----------
model = HeartDiseaseModel(input_dim=X_train.shape[1])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, mode="max", restore_best_weights=True, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    verbose=2
)

# ---------- pick threshold on validation to maximize ACCURACY ----------
val_probs = model.predict(X_val, verbose=0).ravel()

def eval_thr(thr):
    yv = (val_probs >= thr).astype(int)
    return (
        accuracy_score(y_val, yv),
        precision_score(y_val, yv, zero_division=0),
        recall_score(y_val, yv, zero_division=0),
        f1_score(y_val, yv, zero_division=0),
    )

# Stage 1: coarse grid
coarse = np.linspace(0.05, 0.99, 191)  # step 0.005
best_thr, best_acc, best_tuple = 0.5, -1.0, None
for t in coarse:
    acc, prec, rec, f1 = eval_thr(t)
    if acc > best_acc or (acc == best_acc and prec > (best_tuple[1] if best_tuple else -1)):
        best_thr, best_acc, best_tuple = float(t), acc, (acc, prec, rec, f1)

# Stage 2: fine grid around best ±0.05 (clamped to [0.01,0.999])
low = max(0.01, best_thr - 0.05)
high = min(0.999, best_thr + 0.05)
fine = np.linspace(low, high, int((high - low) / 0.001) + 1)

for t in fine:
    acc, prec, rec, f1 = eval_thr(t)
    if acc > best_acc or (acc == best_acc and prec > best_tuple[1]):
        best_thr, best_acc, best_tuple = float(t), acc, (acc, prec, rec, f1)

print(f"\nSelected threshold for accuracy (from validation): {best_thr:.3f} | "
      f"val_acc={best_tuple[0]:.4f}, val_prec={best_tuple[1]:.4f}, val_rec={best_tuple[2]:.4f}, val_f1={best_tuple[3]:.4f}")

with open(os.path.join(DATA_DIR, "selected_threshold.json"), "w") as f:
    json.dump({"threshold": best_thr, "val_metrics": {
        "accuracy": best_tuple[0], "precision": best_tuple[1], "recall": best_tuple[2], "f1": best_tuple[3]
    }}, f, indent=2)

# ---------- save weights ----------
weights_path = os.path.join(DATA_DIR, "best_model.weights.h5")
model.save_weights(weights_path)
print(f"Saved weights to {weights_path}")

# ---------- evaluate on TEST using selected threshold ----------
test_probs = model.predict(X_test, verbose=0).ravel()
y_pred = (test_probs >= best_thr).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, test_probs)

print("\n=== FINAL TEST METRICS (using picked threshold) ===")
print(f"Accuracy : {acc*100:.2f}%")
print(f"AUC      : {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")

# ---------- confusion matrix plot ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
for (i,j),v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.xticks([0,1], ["No CHD", "CHD"])
plt.yticks([0,1], ["No CHD", "CHD"])
plt.title(f"Confusion Matrix (thr={best_thr:.3f})")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "cm_test.png"), dpi=150)
plt.close()

# ---------- training curves ----------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "training_curves.png"), dpi=150)
plt.close()
