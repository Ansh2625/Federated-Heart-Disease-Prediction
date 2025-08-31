import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

RANDOM_STATE = 42

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))          # .../centralized_model
ROOT = os.path.dirname(HERE)                               # project root
DATA_DIR = os.path.join(ROOT, "data")                      # .../data  (sibling of centralized_model)
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "framingham.csv")

# ---------- load ----------
df = pd.read_csv(CSV_PATH)

TARGET = "TenYearCHD"

# ---------- basic cleaning ----------
df = df.drop_duplicates().reset_index(drop=True)

# ---------- feature engineering ----------
df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
df["map_bp"] = df["diaBP"] + (df["sysBP"] - df["diaBP"]) / 3.0

# ---------- split train / val / test ----------
y = df[TARGET].astype(int).values
X = df.drop(columns=[TARGET])

RANDOM_STATE = 7  # <- lock new seed

# test 10%
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE
)
# val 10% of total  (≈11.11% of tmp)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.11111, stratify=y_tmp, random_state=RANDOM_STATE
)


feature_names = X_train.columns.tolist()

# ---------- winsorization on TRAIN ----------
q_low = X_train.quantile(0.01)
q_high = X_train.quantile(0.99)

def apply_clip(df_part):
    return df_part.clip(lower=q_low, upper=q_high, axis=1)

X_train = apply_clip(X_train)
X_val = apply_clip(X_val)
X_test = apply_clip(X_test)

# ---------- impute ----------
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# ---------- scale ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ---------- save ----------
np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train_scaled)
np.save(os.path.join(DATA_DIR, "X_val.npy"),   X_val_scaled)
np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test_scaled)

np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

joblib.dump(imputer, os.path.join(DATA_DIR, "imputer.joblib"))
joblib.dump(scaler,  os.path.join(DATA_DIR, "scaler.joblib"))
np.save(os.path.join(DATA_DIR, "feature_names.npy"), np.array(feature_names))

np.save(os.path.join(DATA_DIR, "idx_train.npy"), X_train.index.values)
np.save(os.path.join(DATA_DIR, "idx_val.npy"),   X_val.index.values)
np.save(os.path.join(DATA_DIR, "idx_test.npy"),  X_test.index.values)

print("Done. Preprocessed train/val/test saved to data/.")
