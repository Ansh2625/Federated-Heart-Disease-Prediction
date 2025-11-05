import os, json, numpy as np, tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from model_2 import SAINT

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

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Positives (train/val/test): {sum(y_train)} / {sum(y_val)} / {sum(y_test)}")

# ===== Model =====
model = SAINT(input_dim=X_train.shape[1], depth=8, dim=256, heads=4, dropout=0.3, ffn_mult=4)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
        tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
    ]
)

# ===== :Differential Privacy =====
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=128,
    learning_rate=1e-4
)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc_roc", curve="ROC"), tf.keras.metrics.AUC(name="auc_pr", curve="PR")]
)

# ===== Callbacks =====
ckpt_path = os.path.join(ART, "saint_best")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc_roc",
                                       save_best_only=True, mode="max",
                                       save_weights_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc_roc", patience=15,
                                     restore_best_weights=True, mode="max", verbose=1),
]

# ===== Training =====
print("\n=== Training SAINT (deep transformer MLP) ===")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

# ===== Evaluation =====
print("\n=== Evaluating on Test Set ===")
model.load_weights(ckpt_path)
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
print(json.dumps(results, indent=2))
with open(os.path.join(ART, "saint_results.json"), "w") as f:
    json.dump(results, f, indent=2)
