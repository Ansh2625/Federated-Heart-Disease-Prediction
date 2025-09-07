import os, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 7

# paths (FIXED: go up ONE level to centralized/)
HERE = os.path.dirname(os.path.abspath(__file__))   
ROOT = os.path.dirname(HERE)                           
RAW  = os.path.join(ROOT, "data", "raw", "framingham.csv")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

# load
df = pd.read_csv(RAW).drop_duplicates().reset_index(drop=True)
TARGET = "TenYearCHD"

# feature engineering
df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
df["map_bp"] = df["diaBP"] + (df["sysBP"] - df["diaBP"]) / 3.0

y = df[TARGET].astype(int).values
X = df.drop(columns=[TARGET])

# 80/10/10 split
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.11111, stratify=y_tmp, random_state=RANDOM_STATE
)

# winsorize (train-only 1/99)
q_low = X_train.quantile(0.01); q_high = X_train.quantile(0.99)
def clip_df(d): return d.clip(lower=q_low, upper=q_high, axis=1)
X_train = clip_df(X_train); X_val = clip_df(X_val); X_test = clip_df(X_test)

# impute + scale (fit on train only)
imp = SimpleImputer(strategy="median")
scl = StandardScaler()
X_train = imp.fit_transform(X_train)
scl.fit(X_train)
X_val  = scl.transform(imp.transform(X_val))
X_test = scl.transform(imp.transform(X_test))

# save
np.save(os.path.join(PROC, "X_train.npy"), X_train.astype(np.float32))
np.save(os.path.join(PROC, "X_val.npy"),   X_val.astype(np.float32))
np.save(os.path.join(PROC, "X_test.npy"),  X_test.astype(np.float32))
np.save(os.path.join(PROC, "y_train.npy"), y_train.astype(np.int64))
np.save(os.path.join(PROC, "y_val.npy"),   y_val.astype(np.int64))
np.save(os.path.join(PROC, "y_test.npy"),  y_test.astype(np.int64))
joblib.dump(imp, os.path.join(PROC, "imputer.joblib"))
joblib.dump(scl, os.path.join(PROC, "scaler.joblib"))
np.save(os.path.join(PROC, "feature_names.npy"), np.array(list(X.columns)))

print("Centralized preprocessing → centralized/data/processed")
