# federated/src/train_fedavg_2.py
import os, json, numpy as np, tensorflow as tf
from model_2 import SAINT
from sklearn.metrics import roc_auc_score

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLIENTS_DIR = os.path.join(ROOT, "data", "clients")
GLOBAL_DIR  = os.path.join(ROOT, "data", "global")
ART_DIR     = os.path.join(ROOT, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

# ---------- hyperparams (light config for CPU) ----------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

CLIENTS = sorted([d for d in os.listdir(CLIENTS_DIR) if d.startswith("client_")])
ROUNDS = 10             # reduced from 25
LOCAL_EPOCHS = 2        # reduced from 5
BATCH_SIZE = 256
LR = 1e-4
CLIENT_FRACTION = 0.6   # pick ~3/5 clients each round

# ---------- data ----------
X_val = np.load(os.path.join(GLOBAL_DIR, "X_val.npy"))
y_val = np.load(os.path.join(GLOBAL_DIR, "y_val.npy"))

sample_X = np.load(os.path.join(CLIENTS_DIR, CLIENTS[0], "X.npy"))
n_features = sample_X.shape[1]

# ---------- model (lighter SAINT for CPU) ----------
global_model = SAINT(input_dim=n_features, depth=4, dim=128, heads=4, dropout=0.3, ffn_mult=2)
# force weight creation before get_weights/set_weights
global_model.build(input_shape=(None, n_features))

def load_client(cid):
    cdir = os.path.join(CLIENTS_DIR, cid)
    return (np.load(os.path.join(cdir, "X.npy")),
            np.load(os.path.join(cdir, "y.npy")))

def client_train(init_weights, X, y):
    m = SAINT(input_dim=X.shape[1], depth=4, dim=128, heads=4, dropout=0.3, ffn_mult=2)
    m.build(input_shape=(None, X.shape[1]))   # build before setting weights
    m.set_weights(init_weights)
    m.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    m.fit(X, y, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return m.get_weights(), len(X)

def weighted_fedavg(weight_sets, sizes):
    total = np.sum(sizes)
    new_w = []
    for params in zip(*weight_sets):
        stacked = np.stack(params, axis=0)                 # (num_clients, ...)
        w = np.tensordot(sizes / total, stacked, axes=(0, 0))
        new_w.append(w)
    return new_w

def eval_auc(model, X, y):
    probs = model.predict(X, batch_size=256, verbose=0).ravel()
    return roc_auc_score(y, probs)

history = {"round": [], "val_auc": []}
best_auc = -1.0

print(f"=== FedAvg (SAINT) | clients={len(CLIENTS)}, rounds={ROUNDS}, local_epochs={LOCAL_EPOCHS} ===")
for r in range(1, ROUNDS + 1):
    np.random.shuffle(CLIENTS)
    k = max(1, int(len(CLIENTS) * CLIENT_FRACTION))
    selected = CLIENTS[:k]

    updates, sizes = [], []
    init_w = global_model.get_weights()
    for cid in selected:
        Xc, yc = load_client(cid)
        w, n = client_train(init_w, Xc, yc)
        updates.append(w); sizes.append(n)

    new_weights = weighted_fedavg(updates, np.array(sizes, dtype=np.float32))
    global_model.set_weights(new_weights)

    # monitor on global val
    auc = eval_auc(global_model, X_val, y_val)
    history["round"].append(r); history["val_auc"].append(float(auc))
    print(f"Round {r:02d}/{ROUNDS}  |  val_auc={auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        global_model.save_weights(os.path.join(ART_DIR, "fedavg_saint_best_2"))
        with open(os.path.join(ART_DIR, "fedavg_best_meta_2.json"), "w") as f:
            json.dump({"best_round": r, "best_val_auc": float(best_auc)}, f, indent=2)

# final save + history
global_model.save_weights(os.path.join(ART_DIR, "fedavg_saint_last_2"))
with open(os.path.join(ART_DIR, "fedavg_history_2.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"[Saved] artifacts/fedavg_saint_best_2  (best AUC={best_auc:.4f})")
print(f"[Saved] artifacts/fedavg_saint_last_2")
print(f"[Saved] artifacts/fedavg_history_2.json")
