import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "framingham.csv")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

print(f"Centralized preprocessing → {PROC}")

# ===== 1) Load raw dataset =====
df = pd.read_csv(RAW)

# Drop rows with missing values (simple cleanup)
df = df.dropna()

# Target column
y = df["TenYearCHD"].values
X = df.drop(columns=["TenYearCHD"]).values

# ===== 2) Train/test/val split =====
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Before SMOTE:")
print("Train:", X_train.shape, "Positives:", sum(y_train))
print("Val  :", X_val.shape, "Positives:", sum(y_val))
print("Test :", X_test.shape, "Positives:", sum(y_test))

# ===== 3) Apply SMOTE only on training set =====
smote = SMOTE(random_state=42, sampling_strategy=0.8)  
# 0.8 means positives will be upsampled until ~80% of negatives
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE (train only):")
print("Train:", X_train.shape, "Positives:", sum(y_train))

# ===== 4) Scale features =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ===== 5) Save numpy arrays =====
np.save(os.path.join(PROC, "X_train.npy"), X_train)
np.save(os.path.join(PROC, "X_val.npy"), X_val)
np.save(os.path.join(PROC, "X_test.npy"), X_test)
np.save(os.path.join(PROC, "y_train.npy"), y_train)
np.save(os.path.join(PROC, "y_val.npy"), y_val)
np.save(os.path.join(PROC, "y_test.npy"), y_test)

print("\nFinal saved shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)
