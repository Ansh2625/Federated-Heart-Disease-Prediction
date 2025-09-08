import os, json, numpy as np, matplotlib.pyplot as plt, joblib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_federated as tff
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from model import HeartDiseaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLIENTS_DIR = os.path.join(ROOT, "data", "clients")
GLOBAL_DIR  = os.path.join(ROOT, "data", "global")
ART         = os.path.join(ROOT, "artifacts")
PLOTS       = os.path.join(ART, "plots")
os.makedirs(ART, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)

SEED = 7
np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------- Tweak safely ----------------
BATCH_SIZE   = 32
CLIENT_LR    = 1e-3
SERVER_LR    = 1.0
NUM_ROUNDS   = 20
LOCAL_EPOCHS = 1
HEADLINE_METRIC = "f1"   # "f1" | "bacc" | "acc"
# ------------------------------------------------

# Global validation split for model selection + calibration
X_val = np.load(os.path.join(GLOBAL_DIR, "X_val.npy"))
y_val = np.load(os.path.join(GLOBAL_DIR, "y_val.npy"))

def load_client_arrays(cid: str):
    cdir = os.path.join(CLIENTS_DIR, cid)
    X = np.load(os.path.join(cdir, "X.npy"))
    y = np.load(os.path.join(cdir, "y.npy"))
    return X, y

client_ids = sorted([d for d in os.listdir(CLIENTS_DIR) if d.startswith("client_")])
print("Clients:", client_ids)

# ----- Compute GLOBAL class imbalance for pos_weight (neg/pos) -----
all_y = []
for cid in client_ids:
    _, y_ = load_client_arrays(cid)
    all_y.append(y_)
all_y = np.concatenate(all_y)
pos = float((all_y == 1).sum())
neg = float((all_y == 0).sum())
pos_weight_val = max(1.0, neg / max(1.0, pos))  # guard divide-by-zero
print(f"Global counts → neg={int(neg)} pos={int(pos)} | pos_weight={pos_weight_val:.3f}")

class WeightedBCE(tf.keras.losses.Loss):
    def __init__(self, pos_weight: float):
        super().__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.pos_weight = tf.constant(pos_weight, dtype=tf.float32)
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)  # from_logits=False (sigmoid in model)
        w = y_true * self.pos_weight + (1.0 - y_true) * 1.0
        return tf.reduce_mean(bce * w)

def make_dataset(X, y):
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int64)))
    ds = ds.shuffle(len(y), seed=SEED).batch(BATCH_SIZE).repeat(LOCAL_EPOCHS)
    return ds

# derive input_spec from one client (must be (features, labels))
_spec_example = make_dataset(*load_client_arrays(client_ids[0])).element_spec

def model_fn():
    km = HeartDiseaseModel(input_dim=X_val.shape[1])
    km.compile(
        optimizer=tf.keras.optimizers.Adam(CLIENT_LR),
        loss=WeightedBCE(pos_weight_val),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR",  name="auc_pr"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return tff.learning.models.from_keras_model(
        keras_model=km,
        input_spec=_spec_example,   # (x, y)
        loss=WeightedBCE(pos_weight_val),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                 tf.keras.metrics.AUC(curve="ROC", name="auc_roc")]
    )

iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(CLIENT_LR),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(SERVER_LR),
)
state = iterative_process.initialize()

def best_thresholds_from_probs(y_true, probs):
    ths = np.linspace(0.05, 0.95, 181)
    best = {"acc":(-1,0.5), "f1":(-1,0.5), "bacc":(-1,0.5), "auc": roc_auc_score(y_true, probs)}
    for t in ths:
        y = (probs >= t).astype(int)
        acc  = accuracy_score(y_true, y)
        f1   = f1_score(y_true, y, zero_division=0)
        bacc = balanced_accuracy_score(y_true, y)
        if acc  > best["acc"][0]:  best["acc"]  = (acc, t)
        if f1   > best["f1"][0]:   best["f1"]   = (f1,  t)
        if bacc > best["bacc"][0]: best["bacc"] = (bacc,t)
    return best

def assign_to_keras(state_obj):
    km = HeartDiseaseModel(input_dim=X_val.shape[1])
    km.compile(optimizer=tf.keras.optimizers.Adam(CLIENT_LR),
               loss=WeightedBCE(pos_weight_val),
               metrics=["accuracy"])
    _ = km(tf.zeros((1, X_val.shape[1]), dtype=tf.float32), training=False)
    weights = iterative_process.get_model_weights(state_obj)
    weights.assign_weights_to(km)
    return km

metrics_history = []
best_val = -1.0
best_pack = None  # (round, metric_value, thr, auc_roc, best_path)

for r in range(1, NUM_ROUNDS + 1):
    federated_data = [make_dataset(*load_client_arrays(cid)) for cid in client_ids]
    result = iterative_process.next(state, federated_data)
    state, metrics = result.state, result.metrics

    # track client training metric if available
    train_auc = 0.0
    try:
        train = metrics["client_work"]["train"]
        train_auc = float(train.get("auc_roc", train.get("binary_accuracy", 0.0)))
    except Exception:
        pass
    metrics_history.append(train_auc)

    # validate current global model on global val
    km = assign_to_keras(state)
    val_raw = km.predict(X_val, verbose=0).ravel()

    # --- Platt calibration on val ---
    calibrator = LogisticRegression(solver="liblinear")
    calibrator.fit(val_raw.reshape(-1,1), y_val)
    val_probs = calibrator.predict_proba(val_raw.reshape(-1,1))[:,1]

    best = best_thresholds_from_probs(y_val, val_probs)
    thr   = best[HEADLINE_METRIC][1]
    value = best[HEADLINE_METRIC][0]
    auc   = best["auc"]

    print(f"Round {r:02d} — train_auc={train_auc:.4f} | val_{HEADLINE_METRIC}={value:.4f} @thr={thr:.3f} | val_AUC={auc:.4f}")

    if value > best_val:
        best_val = value
        # Save best weights + calibrator + thresholds
        best_h5 = os.path.join(ART, "fedavg.best.h5")
        km.save_weights(best_h5)
        joblib.dump(calibrator, os.path.join(ART, "platt_calibrator.joblib"))
        with open(os.path.join(ART, "selected_thresholds.json"), "w") as f:
            json.dump({
                "acc":  {"threshold": float(best["acc"][1])},
                "f1":   {"threshold": float(best["f1"][1])},
                "bacc": {"threshold": float(best["bacc"][1])}
            }, f, indent=2)
        with open(os.path.join(ART, "active_threshold.json"), "w") as f:
            json.dump({"metric": HEADLINE_METRIC, "threshold": float(thr)}, f, indent=2)
        best_pack = (r, value, float(thr), float(auc), best_h5)

# Save final weights (not necessarily best)
km_final = assign_to_keras(state)
final_ckpt  = os.path.join(ART, "fedavg.weights")
final_h5    = os.path.join(ART, "fedavg.weights.h5")
km_final.save_weights(final_ckpt); km_final.save_weights(final_h5)
print(f"\nSaved final weights → {final_ckpt} and {final_h5}")

# Plot history
plt.figure(figsize=(6,4))
plt.plot(metrics_history)
plt.title("Client-train AUC per round")
plt.xlabel("Round"); plt.ylabel("AUC")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "fedavg_training.png"), dpi=150); plt.close()

# Summary
if best_pack:
    r_best, v_best, t_best, auc_best, best_path = best_pack
    print(f"BEST (by {HEADLINE_METRIC} on validation): round={r_best}  {HEADLINE_METRIC}={v_best:.4f} @thr={t_best:.3f}  (AUC={auc_best:.4f})")
    print(f"Saved BEST weights → {best_path}")
    print(f"Active threshold for GUI → {os.path.join(ART, 'active_threshold.json')}")
