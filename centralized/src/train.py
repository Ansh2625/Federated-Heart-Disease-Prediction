import os, json, joblib, numpy as np, tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
from model import HeartDiseaseModel, DeepResidualMLP

# ===== Paths =====
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
ART  = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

# ===== Load data =====
X_train = np.load(os.path.join(PROC, "X_train.npy"))
X_val   = np.load(os.path.join(PROC, "X_val.npy"))
X_test  = np.load(os.path.join(PROC, "X_test.npy"))
y_train = np.load(os.path.join(PROC, "y_train.npy"))
y_val   = np.load(os.path.join(PROC, "y_val.npy"))
y_test  = np.load(os.path.join(PROC, "y_test.npy"))

print(f"Train: {X_train.shape} Val: {X_val.shape} Test: {X_test.shape}")
print(f"Positives (train/val/test): {sum(y_train)} / {sum(y_val)} / {sum(y_test)}")

# ===== Compile helper =====
def compile_model(m):
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, verbose=1),
    # tf.keras.callbacks.EarlyStopping(monitor="val_auc_roc", patience=12, mode="max",
    #                                  restore_best_weights=True, verbose=1),
]

# ===== Train TabNet-like =====
tabnet = HeartDiseaseModel(input_dim=X_train.shape[1])
compile_model(tabnet)
print("\n=== Training TabNet-like ===")
tabnet.fit(X_train, y_train, validation_data=(X_val, y_val),
           epochs=200, batch_size=64, callbacks=callbacks, verbose=2)

# ===== Train DeepResidualMLP =====
mlp = DeepResidualMLP(input_dim=X_train.shape[1])
compile_model(mlp)
print("\n=== Training Deep Residual MLP + SE ===")
mlp.fit(X_train, y_train, validation_data=(X_val, y_val),
        epochs=200, batch_size=64, callbacks=callbacks, verbose=2)

# ===== Validation ensemble =====
val_tabnet = tabnet.predict(X_val, verbose=0).ravel()
val_mlp    = mlp.predict(X_val, verbose=0).ravel()
val_avg    = (val_tabnet + val_mlp) / 2

calibrator = LogisticRegression(solver="liblinear")
calibrator.fit(val_avg.reshape(-1, 1), y_val)
joblib.dump(calibrator, os.path.join(ART, "platt_calibrator.joblib"))

val_probs = calibrator.predict_proba(val_avg.reshape(-1, 1))[:, 1]

def metrics_at_threshold(y_true, p, t):
    pred = (p >= t).astype(int)
    return (
        accuracy_score(y_true, pred),
        precision_score(y_true, pred, zero_division=0),
        recall_score(y_true, pred, zero_division=0),
        f1_score(y_true, pred, zero_division=0),
        balanced_accuracy_score(y_true, pred),
    )

best = {"t": 0.5, "acc": -1, "prec": -1, "rec": -1, "f1": -1, "bacc": -1}
for t in np.linspace(0.05, 0.95, 181):
    acc, prec, rec, f1, bacc = metrics_at_threshold(y_val, val_probs, t)
    if (f1 > best["f1"]) or (f1 == best["f1"] and bacc > best["bacc"]) or \
       (f1 == best["f1"] and bacc == best["bacc"] and rec > best["rec"]):
        best = {"t": float(t), "acc": acc, "prec": prec, "rec": rec, "f1": f1, "bacc": bacc}

best_thr = best["t"]
with open(os.path.join(ART, "selected_threshold.json"), "w") as f:
    json.dump({"threshold": best_thr, "val_metrics": best}, f, indent=2)

print(f"\nSelected threshold (by F1): {best_thr:.3f}")
print("Val @thr:", best)

# ===== Final test evaluation =====
test_tabnet = tabnet.predict(X_test, verbose=0).ravel()
test_mlp    = mlp.predict(X_test, verbose=0).ravel()
test_avg    = (test_tabnet + test_mlp) / 2
test_probs  = calibrator.predict_proba(test_avg.reshape(-1, 1))[:, 1]
y_pred = (test_probs >= best_thr).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, test_probs)
bacc = balanced_accuracy_score(y_test, y_pred)

print("\n=== FINAL TEST METRICS (Ensemble: TabNet + DeepResidualMLP+SE) ===")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Balanced : {bacc*100:.2f}%")
print(f"AUC-ROC  : {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")
