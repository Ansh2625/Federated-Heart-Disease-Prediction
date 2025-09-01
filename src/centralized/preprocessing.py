# src/centralized/preprocessing.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import sys

# import Paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(THIS_DIR), "common"))
from paths import Paths
P = Paths()

RANDOM_STATE = 7  # same seed you locked earlier

CSV_PATH = os.path.join(P.DATA_RAW, "framingham.csv")
df = pd.read_csv(CSV_PATH)

TARGET = "TenYearCHD"
df = df.drop_duplicates().reset_index(drop=True)

# engineered features
df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
df["map_bp"] = df["diaBP"] + (df["sysBP"] - df["diaBP"]) / 3.0

y = df[TARGET].astype(int).values
X = df.drop(columns=[TARGET])

# 80/10/10 split (test first, then val)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.11111, stratify=y_tmp, random_state=RANDOM_STATE
)

# winsorize using train only
q_low = X_train.quantile(0.01)
q_high = X_train.quantile(0.99)
def clip_df(d): return d.clip(lower=q_low, upper=q_high, axis=1)
X_train, X_val, X_test = clip_df(X_train), clip_df(X_val), clip_df(X_test)

# impute + scale on train only
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# save to processed/
np.save(os.path.join(P.DATA_PROCESSED, "X_train.npy"), X_train)
np.save(os.path.join(P.DATA_PROCESSED, "X_val.npy"),   X_val)
np.save(os.path.join(P.DATA_PROCESSED, "X_test.npy"),  X_test)
np.save(os.path.join(P.DATA_PROCESSED, "y_train.npy"), y_train)
np.save(os.path.join(P.DATA_PROCESSED, "y_val.npy"),   y_val)
np.save(os.path.join(P.DATA_PROCESSED, "y_test.npy"),  y_test)

joblib.dump(imputer, os.path.join(P.DATA_PROCESSED, "imputer.joblib"))
joblib.dump(scaler,  os.path.join(P.DATA_PROCESSED, "scaler.joblib"))
np.save(os.path.join(P.DATA_PROCESSED, "feature_names.npy"), np.array(df.drop(columns=[TARGET]).columns.tolist()))
np.save(os.path.join(P.DATA_PROCESSED, "idx_train.npy"), X_tmp.index.values if hasattr(X_tmp, "index") else np.array([]))
np.save(os.path.join(P.DATA_PROCESSED, "idx_val.npy"),   np.array([]))
np.save(os.path.join(P.DATA_PROCESSED, "idx_test.npy"),  np.array([]))

print("✅ Preprocessing complete → data/processed/")
