# centralized/src/preprocessing_2.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data", "raw")
RAW = os.path.join(RAW_DIR, "cardio_train.csv")

PROC_DIR = os.path.join(ROOT, "data", "processed")   # keep exactly this: processed/
ART_DIR  = os.path.join(ROOT, "artifacts")            # <<< FIXED: lowercase only
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

print(f"[Preprocess] reading: {RAW}")

# ---------- load ----------
df = pd.read_csv(RAW, sep=",")
# common Kaggle version sometimes has ';' — handle both
if df.shape[1] == 1 and ";" in df.columns[0]:
    df = pd.read_csv(RAW, sep=";")

# normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# remove obvious bad rows
df = df[(df["ap_hi"].between(50, 250)) & (df["ap_lo"].between(30, 200))]
df = df[(df["height"].between(120, 220)) & (df["weight"].between(30, 250))]

# ---------- feature engineering (12 features) ----------
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
    "age_years", "height_cm", "weight_kg",
    "ap_hi", "ap_lo",
    "chol", "gluc",
    "smoke", "alco", "active",
    "bmi", "pulse_pressure"
]

X = pd.DataFrame({
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
y = df["cardio"].astype(int).values

# ---------- split ----------
X_train, X_temp, y_train, y_temp = train_test_split(
    X.values, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Positives (train/val/test): {int(y_train.sum())} / {int(y_val.sum())} / {int(y_test.sum())}")

# ---------- scale ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ---------- save ----------
np.save(os.path.join(PROC_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROC_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(PROC_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(PROC_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROC_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(PROC_DIR, "y_test.npy"),  y_test)

joblib.dump(scaler, os.path.join(ART_DIR, "cardio_scaler.joblib"))
with open(os.path.join(ART_DIR, "cardio_features.json"), "w") as f:
    json.dump({"features": FEATURES}, f, indent=2)

print("\n[Saved] processed/: X_*.npy, y_*.npy")
print("artifacts/: cardio_scaler.joblib, cardio_features.json")
