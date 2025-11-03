# federated/src/prepare_clients_2.py
import os, json, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ----- config -----
RANDOM_STATE   = 7
NUM_CLIENTS    = 5
NON_IID        = False           # set True to enable Dirichlet split
DIRICHLET_ALPHA = 0.3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW  = os.path.join(os.path.dirname(ROOT), "centralized", "data", "raw", "cardio_train.csv")

CLIENTS_DIR = os.path.join(ROOT, "data", "clients")
GLOBAL_DIR  = os.path.join(ROOT, "data", "global")
os.makedirs(CLIENTS_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

# ----- load (handle , / ;) -----
df = pd.read_csv(RAW, sep=",")
if df.shape[1] == 1 and ";" in df.columns[0]:
    df = pd.read_csv(RAW, sep=";")

# normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# basic cleaning (same as centralized/preprocessing_2.py)
df = df[(df["ap_hi"].between(50, 250)) & (df["ap_lo"].between(30, 200))]
df = df[(df["height"].between(120, 220)) & (df["weight"].between(30, 250))]

# ----- feature engineering (12 features) -----
age_years = df["age"] / 365.25
height_cm = df["height"]
weight_kg = df["weight"]
ap_hi     = df["ap_hi"]
ap_lo     = df["ap_lo"]
chol      = df["cholesterol"] if "cholesterol" in df else df["chol"]
gluc      = df["gluc"]
smoke     = df["smoke"]
alco      = df["alco"]
active    = df["active"]

bmi = weight_kg / (height_cm / 100.0) ** 2
pulse_pressure = ap_hi - ap_lo

FEATURES = [
    "age_years","height_cm","weight_kg",
    "ap_hi","ap_lo",
    "chol","gluc",
    "smoke","alco","active",
    "bmi","pulse_pressure"
]

X_all = pd.DataFrame({
    "age_years": age_years,
    "height_cm": height_cm,
    "weight_kg": weight_kg,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "chol": chol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "bmi": bmi,
    "pulse_pressure": pulse_pressure,
})
y_all = df["cardio"].astype(int).values

# persist feature order for GUI / inference
with open(os.path.join(GLOBAL_DIR, "feature_columns.json"), "w") as f:
    json.dump(FEATURES, f, indent=2)

# ----- global splits (10% test, ~10% val of remaining) -----
X_tmp, X_test_raw, y_tmp, y_test = train_test_split(
    X_all, y_all, test_size=0.10, stratify=y_all, random_state=RANDOM_STATE
)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.11111, stratify=y_tmp, random_state=RANDOM_STATE
)

# ----- shuffle training for client partition -----
rng = np.random.RandomState(RANDOM_STATE)
Xr, yr = shuffle(X_train_raw.values, y_train, random_state=RANDOM_STATE)

# ----- partition into clients -----
if NON_IID:
    classes = np.unique(yr)
    idx_by_class = {c: np.where(yr == c)[0] for c in classes}
    for c in classes: rng.shuffle(idx_by_class[c])
    client_idx = [[] for _ in range(NUM_CLIENTS)]
    for c in classes:
        n_c = len(idx_by_class[c])
        props = rng.dirichlet(alpha=np.ones(NUM_CLIENTS) * DIRICHLET_ALPHA)
        counts = (np.floor(props * n_c)).astype(int)
        while counts.sum() < n_c:
            counts[rng.randint(0, NUM_CLIENTS)] += 1
        start = 0
        for cid in range(NUM_CLIENTS):
            end = start + counts[cid]
            client_idx[cid].extend(idx_by_class[c][start:end].tolist()); start = end
    chunks = [np.array(rng.permutation(ix), dtype=int) for ix in client_idx]
else:
    idx = np.arange(len(yr)); rng.shuffle(idx)
    chunks = [chunk for chunk in np.array_split(idx, NUM_CLIENTS)]

def winsor(dfp, ql, qh): return dfp.clip(lower=ql, upper=qh, axis=1)

# ----- per-client local preprocessing (winsor 1–99, median impute, standardize) -----
for i, idx in enumerate(chunks, start=1):
    cdir = os.path.join(CLIENTS_DIR, f"client_{i}")
    os.makedirs(cdir, exist_ok=True)
    Xc_raw = pd.DataFrame(Xr[idx], columns=FEATURES)
    yc     = yr[idx]

    ql = Xc_raw.quantile(0.01); qh = Xc_raw.quantile(0.99)
    Xc_w = winsor(Xc_raw, ql, qh)

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    Xc  = scl.fit_transform(imp.fit_transform(Xc_w))

    np.save(os.path.join(cdir, "X.npy"), Xc.astype(np.float32))
    np.save(os.path.join(cdir, "y.npy"), yc.astype(np.int64))
    joblib.dump(imp, os.path.join(cdir, "imputer.joblib"))
    joblib.dump(scl, os.path.join(cdir, "scaler.joblib"))
    ql.to_json(os.path.join(cdir, "winsor_low.json"))
    qh.to_json(os.path.join(cdir, "winsor_high.json"))

    print(f"client_{i}: n={Xc.shape[0]}  pos={(yc==1).sum()}  neg={(yc==0).sum()}")

# ----- global preprocessors (fit on TRAIN ONLY) applied to val/test -----
qlg = X_train_raw.quantile(0.01); qhg = X_train_raw.quantile(0.99)
def apply_g(d): return d.clip(lower=qlg, upper=qhg, axis=1)

imp_g = SimpleImputer(strategy="median")
scl_g = StandardScaler()
Xt_imp = imp_g.fit_transform(apply_g(X_train_raw.copy()))
scl_g.fit(Xt_imp)

Xv = scl_g.transform(imp_g.transform(apply_g(X_val_raw.copy())))
Xs = scl_g.transform(imp_g.transform(apply_g(X_test_raw.copy())))
np.save(os.path.join(GLOBAL_DIR, "X_val.npy"),  Xv.astype(np.float32))
np.save(os.path.join(GLOBAL_DIR, "X_test.npy"), Xs.astype(np.float32))
np.save(os.path.join(GLOBAL_DIR, "y_val.npy"),  y_val.astype(np.int64))
np.save(os.path.join(GLOBAL_DIR, "y_test.npy"), y_test.astype(np.int64))

# save global preprocessors
joblib.dump(imp_g, os.path.join(GLOBAL_DIR, "global_imputer.joblib"))
joblib.dump(scl_g, os.path.join(GLOBAL_DIR, "global_scaler.joblib"))
qlg.to_json(os.path.join(GLOBAL_DIR, "winsor_low.json"))
qhg.to_json(os.path.join(GLOBAL_DIR, "winsor_high.json"))

print("\nClients → federated/data/clients")
print("Global val/test + preprocessors → federated/data/global")
