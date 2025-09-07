import os, json, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from model import HeartDiseaseModel

# paths (FIXED)
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

# model
model = HeartDiseaseModel(input_dim=X_train.shape[1])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])

callbacks = [
  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
  tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, mode="max",
                                   restore_best_weights=True, verbose=1),
]

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=200, batch_size=64,
                    callbacks=callbacks, verbose=2)

# threshold search
val_probs = model.predict(X_val, verbose=0).ravel()
def eval_thr(t):
    yv = (val_probs >= t).astype(int)
    return (accuracy_score(y_val, yv),
            precision_score(y_val, yv, zero_division=0),
            recall_score(y_val, yv, zero_division=0),
            f1_score(y_val, yv, zero_division=0))
coarse = np.linspace(0.05, 0.99, 191)
best_thr, best_acc, best_tuple = 0.5, -1.0, None
for t in coarse:
    acc, prec, rec, f1 = eval_thr(t)
    if acc > best_acc or (acc == best_acc and prec > (best_tuple[1] if best_tuple else -1)):
        best_thr, best_acc, best_tuple = float(t), acc, (acc, prec, rec, f1)
low, high = max(0.01, best_thr-0.05), min(0.999, best_thr+0.05)
fine = np.linspace(low, high, int((high-low)/0.001)+1)
for t in fine:
    acc, prec, rec, f1 = eval_thr(t)
    if acc > best_acc or (acc == best_acc and prec > best_tuple[1]):
        best_thr, best_acc, best_tuple = float(t), acc, (acc, prec, rec, f1)

with open(os.path.join(ART, "selected_threshold.json"), "w") as f:
    json.dump({"threshold": best_thr, "val_metrics": {
        "accuracy": best_tuple[0], "precision": best_tuple[1],
        "recall": best_tuple[2], "f1": best_tuple[3]}}, f, indent=2)

model.save_weights(os.path.join(ART, "best_model.weights.h5"))

# test eval
test_probs = model.predict(X_test, verbose=0).ravel()
y_pred = (test_probs >= best_thr).astype(int)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, test_probs)

print("\n=== FINAL TEST METRICS (centralized) ===")
print(f"Accuracy : {acc*100:.2f}%")
print(f"AUC      : {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")

# plots
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4)); plt.imshow(cm, cmap="Blues")
for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
plt.xticks([0,1],["No CHD","CHD"]); plt.yticks([0,1],["No CHD","CHD"])
plt.title(f"Confusion Matrix (thr={best_thr:.3f})"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"test_cm.png"), dpi=150); plt.close()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"]); plt.title("Accuracy"); plt.legend(["train","val"])
plt.subplot(1,2,2); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"]); plt.title("Loss"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"training_curves.png"), dpi=150); plt.close()
