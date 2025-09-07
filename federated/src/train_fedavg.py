import os, json, numpy as np, matplotlib.pyplot as plt
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Optional: uncomment to silence CUDA plugin noise on CPU-only machines
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_federated as tff
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from model import HeartDiseaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLIENTS_DIR = os.path.join(ROOT, "data", "clients")
GLOBAL_DIR  = os.path.join(ROOT, "data", "global")
ART         = os.path.join(ROOT, "artifacts")
PLOTS       = os.path.join(ART, "plots")
os.makedirs(ART, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

SEED = 7
np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------- You can tweak these safely ----------------
BATCH_SIZE   = 32
CLIENT_LR    = 1e-3
SERVER_LR    = 1.0
NUM_ROUNDS   = 20
LOCAL_EPOCHS = 1
HEADLINE_METRIC = "acc"  # acc | f1 | bacc  (what you want to show)
# ------------------------------------------------------------

# Load global val/test (we only need val during training)
X_val  = np.load(os.path.join(GLOBAL_DIR, "X_val.npy"))
y_val  = np.load(os.path.join(GLOBAL_DIR, "y_val.npy"))

def load_client_arrays(cid: str):
    cdir = os.path.join(CLIENTS_DIR, cid)
    X = np.load(os.path.join(cdir, "X.npy"))
    y = np.load(os.path.join(cdir, "y.npy"))
    return X, y

def make_dataset(X, y):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(y), seed=SEED).batch(BATCH_SIZE).repeat(LOCAL_EPOCHS)
    return ds

client_ids = sorted([d for d in os.listdir(CLIENTS_DIR) if d.startswith("client_")])
print("Clients:", client_ids)

def model_fn():
    km = HeartDiseaseModel(input_dim=X_val.shape[1])
    km.compile(
        optimizer=tf.keras.optimizers.Adam(CLIENT_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                 tf.keras.metrics.AUC(curve="ROC", name="auc")],
    )
    # Input spec from some client
    spec = make_dataset(*load_client_arrays(client_ids[0])).element_spec
    return tff.learning.models.from_keras_model(
        keras_model=km,
        input_spec=spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                 tf.keras.metrics.AUC(curve="ROC", name="auc")],
    )

iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(CLIENT_LR),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(SERVER_LR),
)
state = iterative_process.initialize()

# Utilities ---------------------------------------------------
def extract_train_acc(m):
    try:
        train = m["client_work"]["train"]
        # prefer auc if present for sanity, fall back to accuracy
        if "auc" in train:
            return float(train["auc"])
        for k, v in train.items():
            if "accuracy" in k:
                return float(v)
    except Exception:
        pass
    return 0.0

def best_thresholds_from_probs(y_true, probs):
    ths = np.linspace(0.05, 0.95, 181)
    best = {"acc":(-1,0.5), "f1":(-1,0.5), "bacc":(-1,0.5), "auc": roc_auc_score(y_true, probs)}
    for t in ths:
        y = (probs >= t).astype(int)
        acc  = accuracy_score(y_true, y)
        f1   = f1_score(y_true, y, zero_division=0)
        rec  = recall_score(y_true, y, zero_division=0)
        prec = precision_score(y_true, y, zero_division=0)
        bacc = balanced_accuracy_score(y_true, y)
        if acc  > best["acc"][0]:  best["acc"]  = (acc, t)
        if f1   > best["f1"][0]:   best["f1"]   = (f1,  t)
        if bacc > best["bacc"][0]: best["bacc"] = (bacc,t)
    return best

def assign_to_keras(state_obj):
    km = HeartDiseaseModel(input_dim=X_val.shape[1])
    km.compile(optimizer=tf.keras.optimizers.Adam(CLIENT_LR),
               loss="binary_crossentropy",
               metrics=["accuracy"])
    _ = km(tf.zeros((1, X_val.shape[1]), dtype=tf.float32), training=False)
    weights = iterative_process.get_model_weights(state_obj)
    weights.assign_weights_to(km)
    return km
# ------------------------------------------------------------

metrics_history = []
best_val = -1.0
best_pack = None  # (round, metric_value, thr, auc, weights_path)

for r in range(1, NUM_ROUNDS + 1):
    # one FL round
    federated_data = [make_dataset(*load_client_arrays(cid)) for cid in client_ids]
    result = iterative_process.next(state, federated_data)
    state, metrics = result.state, result.metrics
    tr = extract_train_acc(metrics)
    metrics_history.append(tr)

    # build keras model for validation and compute headline metric with its own best threshold
    km = assign_to_keras(state)
    val_probs = km.predict(X_val, verbose=0).ravel()
    best = best_thresholds_from_probs(y_val, val_probs)
    thr   = best[HEADLINE_METRIC][1]
    value = best[HEADLINE_METRIC][0]
    auc   = best["auc"]

    print(f"Round {r:02d} — train={tr:.4f} | val_{HEADLINE_METRIC}={value:.4f} @thr={thr:.3f} | val_AUC={auc:.4f}")

    # keep only best-by-headline
    if value > best_val:
        best_val = value
        # save weights + thresholds NOW
        best_h5 = os.path.join(ART, "fedavg.best.h5")
        km.save_weights(best_h5)
        with open(os.path.join(ART, "selected_thresholds.json"), "w") as f:
            json.dump({
                "acc":  {"threshold": float(best["acc"][1])},
                "f1":   {"threshold": float(best["f1"][1])},
                "bacc": {"threshold": float(best["bacc"][1])}
            }, f, indent=2)
        # also persist the ACTIVE (GUI) threshold as your headline
        with open(os.path.join(ART, "active_threshold.json"), "w") as f:
            json.dump({"metric": HEADLINE_METRIC, "threshold": float(thr)}, f, indent=2)
        best_pack = (r, value, float(thr), float(auc), best_h5)

# After rounds: also save the final weights (not necessarily best)
final_model = assign_to_keras(state)
final_ckpt  = os.path.join(ART, "fedavg.weights")
final_h5    = os.path.join(ART, "fedavg.weights.h5")
final_model.save_weights(final_ckpt)
final_model.save_weights(final_h5)
print(f"\nSaved final weights → {final_ckpt} and {final_h5}")

# Summary + plot
r_best, v_best, t_best, auc_best, best_path = best_pack
print(f"BEST (by {HEADLINE_METRIC} on validation): round={r_best}  {HEADLINE_METRIC}={v_best:.4f} @thr={t_best:.3f}  (AUC={auc_best:.4f})")
print(f"Saved BEST weights → {best_path}")
print(f"Active threshold for GUI → {os.path.join(ART, 'active_threshold.json')}")

plt.figure(figsize=(6,4))
plt.plot(metrics_history)
plt.title("Train metric per round (client agg)")
plt.xlabel("Round"); plt.ylabel("Train metric (AUC or Acc)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "fedavg_training.png"), dpi=150)
plt.close()
