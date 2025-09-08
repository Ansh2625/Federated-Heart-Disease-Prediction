import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU, avoids GPU warnings

import os, sys, json
from typing import Dict, Tuple, Callable

import numpy as np

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTDIR = os.path.join(ROOT, "centralized", "artifacts")
PLOTS  = os.path.join(ARTDIR, "plots")
DATAD  = os.path.join(ROOT, "centralized", "data", "processed")

# --- UI schema & sample ---
def get_schema():
    return [
        {"name": "male",             "label": "Male (0/1)",               "type": "bool"},
        {"name": "age",              "label": "Age (years)",              "type": "int"},
        {"name": "education",        "label": "Education (1-4)",          "type": "int"},
        {"name": "currentSmoker",    "label": "Current Smoker (0/1)",     "type": "bool"},
        {"name": "cigsPerDay",       "label": "Cigs per Day",             "type": "float"},
        {"name": "BPMeds",           "label": "BP Meds (0/1)",            "type": "bool"},
        {"name": "prevalentStroke",  "label": "Prev. Stroke (0/1)",       "type": "bool"},
        {"name": "prevalentHyp",     "label": "Prev. Hypertension (0/1)", "type": "bool"},
        {"name": "diabetes",         "label": "Diabetes (0/1)",           "type": "bool"},
        {"name": "totChol",          "label": "Total Cholesterol",        "type": "float"},
        {"name": "sysBP",            "label": "Systolic BP",              "type": "float"},
        {"name": "diaBP",            "label": "Diastolic BP",             "type": "float"},
        {"name": "BMI",              "label": "BMI",                      "type": "float"},
        {"name": "heartRate",        "label": "Heart Rate",               "type": "float"},
        {"name": "glucose",          "label": "Glucose",                  "type": "float"},
    ]

def get_sample_input() -> Dict[str, float]:
    return {
        "male": 0, "age": 45, "education": 2, "currentSmoker": 0,
        "cigsPerDay": 0.0, "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0,
        "diabetes": 0, "totChol": 200.0, "sysBP": 120.0, "diaBP": 80.0,
        "BMI": 24.0, "heartRate": 75.0, "glucose": 90.0
    }

# --- helpers for preprocessing ---
def _load_feature_order() -> list:
    # you saved this in preprocessing: feature_names.npy
    p = os.path.join(DATAD, "feature_names.npy")
    if not os.path.isfile(p):
        raise FileNotFoundError("Missing centralized/data/processed/feature_names.npy")
    arr = np.load(p, allow_pickle=True)
    return list(arr.tolist())

def _vectorize(payload: Dict[str, float], feature_order: list) -> np.ndarray:
    # GUI fields match your raw column names
    row = [payload.get(k, 0.0) for k in feature_order]
    return np.array(row, dtype=float).reshape(1, -1)

def _load_imputer_scaler():
    from joblib import load
    imp_p = os.path.join(DATAD, "imputer.joblib")
    scl_p = os.path.join(DATAD, "scaler.joblib")
    if not os.path.isfile(imp_p) or not os.path.isfile(scl_p):
        raise FileNotFoundError("Missing imputer.joblib/scaler.joblib in centralized/data/processed")
    return load(imp_p), load(scl_p)

def _apply_preprocess(x: np.ndarray) -> np.ndarray:
    imputer, scaler = _load_imputer_scaler()
    x_imp = imputer.transform(x)
    x_scl = scaler.transform(x_imp)
    return x_scl

# --- model loading ---
def _try_load_sklearn() -> Tuple[object, str] | None:
    for f in sorted(os.listdir(ARTDIR), key=lambda n: n.lower()):
        if f.lower().endswith((".joblib", ".pkl")):
            from joblib import load
            return load(os.path.join(ARTDIR, f)), f
    return None

def _is_valid_keras_h5(path: str) -> bool:
    try:
        import h5py
        with h5py.File(path, "r") as h:
            if "model_config" in h or "layer_names" in h:
                return True
            for k in ("keras_version","backend","model_config"):
                if k in h.attrs: return True
        return False
    except Exception:
        return False

def _try_load_full_keras() -> tuple | None:
    for f in sorted(os.listdir(ARTDIR), key=lambda n: n.lower()):
        fl = f.lower()
        if fl.endswith((".keras", ".h5")):
            try:
                import tensorflow as tf
                m = tf.keras.models.load_model(os.path.join(ARTDIR, f), compile=False)
                return m, f
            except Exception:
                continue
    return None


def _build_model_from_weights(input_dim: int) -> Tuple[object, str] | None:
    """
    If only weights exist (e.g., best_model.weights.h5), we import centralized/src/model.py
    and try common builder names to get the architecture, then load_weights.
    """
    weights_candidates = [f for f in os.listdir(ARTDIR) if f.lower().endswith(".weights.h5")]
    if not weights_candidates:
        # also allow generic .h5 files that are weights-only
        weights_candidates = [f for f in os.listdir(ARTDIR) if f.lower().endswith(".h5")]
    if not weights_candidates:
        return None

    # import your model module
    sys.path.append(os.path.join(ROOT, "centralized", "src"))
    try:
        import importlib
        mdl = importlib.import_module("model")
    except Exception as e:
        raise FileNotFoundError("centralized/src/model.py not importable, needed to rebuild model from weights.") from e

    # guess builder function
    candidates = ["build_model", "create_model", "get_model", "make_model", "build_baseline_model"]
    net = None
    for name in candidates:
        fn: Callable | None = getattr(mdl, name, None)
        if fn is None:
            continue
        try:
            # Try signatures with input_dim / input_shape
            try:
                net = fn(input_dim=input_dim)
            except TypeError:
                try:
                    net = fn(input_shape=(input_dim,))
                except TypeError:
                    net = fn()  # maybe reads global config
            break
        except Exception:
            continue

    if net is None:
        raise RuntimeError("Could not build model from centralized/src/model.py (no suitable builder function).")

    import tensorflow as tf
    for wf in sorted(weights_candidates):
        wpath = os.path.join(ARTDIR, wf)
        try:
            net.load_weights(wpath)
            return net, wf
        except Exception:
            continue
    return None

def _load_predictor(input_dim: int) -> Tuple[Callable[[np.ndarray], float], str]:
    """
    Returns (predict_fn, model_name). predict_fn expects preprocessed X and returns prob in [0,1].
    Tries sklearn -> full keras -> keras-from-weights (using model.py).
    """
    # sklearn pipeline/regressor/classifier
    s = _try_load_sklearn()
    if s:
        model, name = s
        if hasattr(model, "predict_proba"):
            return (lambda X: float(model.predict_proba(X)[0, 1])), name
        else:
            return (lambda X: float(np.clip(model.predict(X)[0], 0.0, 1.0))), name

    # full keras model
    k = _try_load_full_keras()
    if k:
        model, name = k
        def _kpred(X):
            y = model.predict(X, verbose=0)
            val = float(y[0] if hasattr(y, "__len__") else y)
            if val < 0 or val > 1:
                import math
                val = 1/(1+math.exp(-val))
            return val
        return _kpred, name

    # weights-only -> rebuild via your model.py
    w = _build_model_from_weights(input_dim)
    if w:
        model, name = w
        def _wpred(X):
            y = model.predict(X, verbose=0)
            val = float(y[0] if hasattr(y, "__len__") else y)
            if val < 0 or val > 1:
                import math
                val = 1/(1+math.exp(-val))
            return val
        return _wpred, name

    raise FileNotFoundError(
        "Found files but none are valid models. "
        "Tip: save a sklearn pipeline as .joblib/.pkl or a full Keras model (.keras or model.h5), "
        "or keep weights as *.weights.h5 and ensure centralized/src/model.py exposes a builder."
    )

# --- public API for GUI ---
def predict(payload: Dict, log=print) -> Dict:
    feats = _load_feature_order()
    x = _vectorize(payload, feats)
    x = _apply_preprocess(x)
    pred_fn, model_name = _load_predictor(input_dim=x.shape[1])
    proba = float(pred_fn(x))
    proba = float(np.clip(proba, 0.0, 1.0))
    label = int(proba >= 0.5)

    # optional threshold if you saved one
    thr_file = os.path.join(ARTDIR, "selected_threshold.json")
    if os.path.isfile(thr_file):
        try:
            thr = float(json.load(open(thr_file)).get("threshold", 0.5))
            label = int(proba >= thr)
        except Exception:
            pass

    return {
        "proba": proba,
        "proba_text": f"Risk: {proba*100:.2f}%",
        "label": label,
        "extra": f"Model: {model_name}"
    }

def get_plots_dir():
    return PLOTS
