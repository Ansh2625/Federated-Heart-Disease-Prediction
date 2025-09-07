# federated/src/train_fedavg.py
import os, json, numpy as np, matplotlib.pyplot as plt

# Quiet TF logs (set BEFORE importing TF)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # avoid GPU probes on CPU-only

import tensorflow as tf
import tensorflow_federated as tff
from model import HeartDiseaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Paths & dirs
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLIENTS_DIR = os.path.join(ROOT, "data", "clients")
GLOBAL_DIR  = os.path.join(ROOT, "data", "global")
ART   = os.path.join(ROOT, "artifacts")
PLOTS = os.path.join(ART, "plots")
os.makedirs(ART, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

# Hyperparams
BATCH_SIZE = 32
CLIENT_LR  = 1e-3
SERVER_LR  = 1.0
NUM_ROUNDS = 20

# Load global val/test
X_val  = np.load(os.path.join(GLOBAL_DIR, "X_val.npy"))
y_val  = np.load(os.path.join(GLOBAL_DIR, "y_val.npy"))
X_test = np.load(os.path.join(GLOBAL_DIR, "X_test.npy"))
y_test = np.load(os.path.join(GLOBAL_DIR, "y_test.npy"))

# Client data loader
def load_client_data(cid: str) -> tf.data.Dataset:
    cdir = os.path.join(CLIENTS_DIR, cid)
    X = np.load(os.path.join(cdir, "X.npy"))
    y = np.load(os.path.join(cdir, "y.npy"))
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(y)).batch(BATCH_SIZE)
    return ds

client_ids = sorted([d for d in os.listdir(CLIENTS_DIR) if d.startswith("client_")])
print("Clients:", client_ids)

# TFF model function
def model_fn():
    keras_model = HeartDiseaseModel(input_dim=X_val.shape[1])
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(CLIENT_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=load_client_data(client_ids[0]).element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )

# Federated algorithm
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(CLIENT_LR),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(SERVER_LR),
)

state = iterative_process.initialize()

# Training loop
metrics_history = []

def _extract_train_acc(m):
    try:
        train = m["client_work"]["train"]
        for k, v in train.items():
            if "accuracy" in k:
                return float(v)
    except Exception:
        pass
    return 0.0

for round_num in range(1, NUM_ROUNDS + 1):
    federated_data = [load_client_data(cid) for cid in client_ids]
    result = iterative_process.next(state, federated_data)
    state, metrics = result.state, result.metrics
    acc = _extract_train_acc(metrics)
    metrics_history.append(acc)
    print(f"Round {round_num:02d} — train acc={acc:.4f}")

# Export final global model
final_model = HeartDiseaseModel(input_dim=X_val.shape[1])
final_model.compile(optimizer=tf.keras.optimizers.Adam(CLIENT_LR),
                    loss="binary_crossentropy", metrics=["accuracy"])

# >>> BUILD THE MODEL BEFORE ASSIGNING/SAVING <<<
_ = final_model(tf.zeros((1, X_val.shape[1]), dtype=tf.float32), training=False)

weights = iterative_process.get_model_weights(state)
weights.assign_weights_to(final_model)

# Save in BOTH formats for max compatibility
WEIGHTS_TF = os.path.join(ART, "fedavg.weights")     # TF Checkpoint format
WEIGHTS_H5 = os.path.join(ART, "fedavg.weights.h5")  # H5 format
final_model.save_weights(WEIGHTS_TF)
final_model.save_weights(WEIGHTS_H5)
print(f"Saved federated weights → {WEIGHTS_TF}  and  {WEIGHTS_H5}")

# Threshold search on validation (accuracy-based; switch to F1 if you prefer)
val_probs = final_model.predict(X_val, verbose=0).ravel()

def eval_thr(t):
    yv = (val_probs >= t).astype(int)
    return (
        accuracy_score(y_val, yv),
        precision_score(y_val, yv, zero_division=0),
        recall_score(y_val, yv, zero_division=0),
        f1_score(y_val, yv, zero_division=0),
    )

best_thr, best_acc, best_tuple = 0.5, -1, None
for t in np.linspace(0.05, 0.99, 191):
    acc, prec, rec, f1 = eval_thr(t)
    if acc > best_acc:
        best_thr, best_acc, best_tuple = t, acc, (acc, prec, rec, f1)

with open(os.path.join(ART, "selected_threshold.json"), "w") as f:
    json.dump(
        {
            "threshold": float(best_thr),
            "val_metrics": {
                "accuracy": best_tuple[0],
                "precision": best_tuple[1],
                "recall": best_tuple[2],
                "f1": best_tuple[3],
            },
        },
        f,
        indent=2,
    )

print(f"\nSelected threshold={best_thr:.3f} | val_acc={best_tuple[0]:.4f}")

# Training curve plot
plt.figure(figsize=(6, 4))
plt.plot(metrics_history)
plt.title("FedAvg Training Accuracy (per round)")
plt.xlabel("Round")
plt.ylabel("Train Acc")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "fedavg_training.png"), dpi=150)
plt.close()
