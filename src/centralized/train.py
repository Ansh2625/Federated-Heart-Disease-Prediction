# src/centralized/train.py
import os, sys, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# paths helper
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(THIS_DIR), "common"))
from paths import Paths
P = Paths()

# local import
from model import HeartDiseaseModel

# load processed arrays
X_train = np.load(os.path.join(P.DATA_PROCESSED, "X_train.npy"))
X_val   = np.load(os.path.join(P.DATA_PROCESSED, "X_val.npy"))
X_test  = np.load(os.path.join(P.DATA_PROCESSED, "X_test.npy"))
y_train = np.load(os.path.join(P.DATA_PROCESSED, "y_train.npy"))
y_val   = np.load(os.path.join(P.DATA_PROCESSED, "y_val.npy"))
y_test  = np.load(os.path.join(P.DATA_PROCESSED, "y_test.npy"))

print("Class distribution (train):", np.bincount(y_train))

model = HeartDiseaseModel(input_dim=X_train.shape[1])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, mode="max", restore_best_weights=True, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200, batch_size=64, callbacks=callbacks, verbose=2
)

# threshold search (maximize validation accuracy; tie-break by precision)
val_probs = model.predict(X_val, verbose=0).ravel()
def eval_thr(t):
    yv = (val_probs >= t).astype(int)
    return (accuracy_score(y_val, yv),
            precision_score(y_val, yv, zero_division=0),
            recall_score(y_val, yv, zero_division=0),
            f1_score(y_val, yv, zero_division=0))
best_thr, best_acc, best_tuple = 0.5, -1.0, None
for t in np.linspace(0.05, 0.95, 901):
    acc, prec, rec, f1 = eval_thr(t)
    if acc > best_acc or (acc == best_acc and prec > (best_tuple[1] if best_tuple else -1)):
        best_thr, best_acc, best_tuple = float(t), acc, (acc, prec, rec, f1)

# save weights & threshold to artifacts/centralized
model.save_weights(os.path.join(P.ARTIFACTS_CENT, "best_model.weights.h5"))
with open(os.path.join(P.ARTIFACTS_CENT, "selected_threshold.json"), "w") as f:
    json.dump({"threshold": best_thr,
               "val_metrics": {"accuracy": best_tuple[0], "precision": best_tuple[1], "recall": best_tuple[2], "f1": best_tuple[3]}},
              f, indent=2)

# test evaluation
test_probs = model.predict(X_test, verbose=0).ravel()
y_pred = (test_probs >= best_thr).astype(int)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, test_probs)

print("\n=== FINAL TEST METRICS ===")
print(f"Accuracy : {acc*100:.2f}%  | AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
cm = confusion_matrix(y_test, y_pred)

# plots to artifacts/centralized/plots
plots = P.PLOTS_CENT
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
for (i,j),v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.xticks([0,1], ["No CHD","CHD"]); plt.yticks([0,1], ["No CHD","CHD"])
plt.title(f"Confusion Matrix (thr={best_thr:.3f})")
plt.tight_layout(); plt.savefig(os.path.join(plots, "test_cm.png"), dpi=150); plt.close()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"]); plt.title("Accuracy"); plt.legend(["train","val"])
plt.subplot(1,2,2); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"]); plt.title("Loss"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig(os.path.join(plots, "training_curves.png"), dpi=150); plt.close()
