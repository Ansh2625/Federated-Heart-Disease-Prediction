import os, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 7
NUM_CLIENTS = 5
NON_IID = False       # keep False now; we can flip later
DIRICHLET_ALPHA = 0.3 # used only when NON_IID=True

# paths (local to the federated tree)
HERE = os.path.dirname(os.path.abspath(__file__))    
ROOT = os.path.dirname(HERE)                   
RAW  = os.path.join(os.path.dirname(ROOT), "centralized", "data", "raw", "framingham.csv")
CLIENTS_DIR = os.path.join(ROOT, "data", "clients")
GLOBAL_DIR  = os.path.join(ROOT, "data", "global")
os.makedirs(CLIENTS_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

# load & feature engineering 
df = pd.read_csv(RAW).drop_duplicates().reset_index(drop=True)
df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
df["map_bp"] = df["diaBP"] + (df["sysBP"] - df["diaBP"]) / 3.0
TARGET = "TenYearCHD"
y_all = df[TARGET].astype(int).values
X_all = df.drop(columns=[TARGET])

# keep the same global split as centralized (80/10/10)
X_tmp, X_test_raw, y_tmp, y_test = train_test_split(
    X_all, y_all, test_size=0.10, stratify=y_all, random_state=RANDOM_STATE
)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.11111, stratify=y_tmp, random_state=RANDOM_STATE
)

#  partition TRAIN into 5 clients 
rng = np.random.RandomState(RANDOM_STATE)
Xr, yr = shuffle(X_train_raw.values, y_train, random_state=RANDOM_STATE)

if NON_IID:
    # label-skew via Dirichlet (optional, off by default)
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
    # IID equal chunks
    idx = np.arange(len(yr)); rng.shuffle(idx)
    chunks = [chunk for chunk in np.array_split(idx, NUM_CLIENTS)]

def winsor(dfp, ql, qh): return dfp.clip(lower=ql, upper=qh, axis=1)

#  per-client LOCAL preprocessing 
for i, idx in enumerate(chunks, start=1):
    cdir = os.path.join(CLIENTS_DIR, f"client_{i}")
    os.makedirs(cdir, exist_ok=True)

    Xc_raw = pd.DataFrame(Xr[idx], columns=X_train_raw.columns)
    yc     = yr[idx]

    # winsorize using THIS CLIENT ONLY (1%/99%), impute & scale using THIS CLIENT ONLY
    ql = Xc_raw.quantile(0.01); qh = Xc_raw.quantile(0.99)
    Xc_w = winsor(Xc_raw, ql, qh)
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    Xc  = scl.fit_transform(imp.fit_transform(Xc_w))

    # save client arrays + its local preprocessors
    np.save(os.path.join(cdir, "X.npy"), Xc.astype(np.float32))
    np.save(os.path.join(cdir, "y.npy"), yc.astype(np.int64))
    joblib.dump(imp, os.path.join(cdir, "imputer.joblib"))
    joblib.dump(scl, os.path.join(cdir, "scaler.joblib"))
    ql.to_json(os.path.join(cdir, "winsor_low.json"))
    qh.to_json(os.path.join(cdir, "winsor_high.json"))
    print(f"client_{i}: n={Xc.shape[0]}  pos={int((yc==1).sum())}  neg={int((yc==0).sum())}")

#  server-side global val/test (for evaluation only)
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

print("\nClients → federated/data/clients")
print("Global val/test → federated/data/global")
