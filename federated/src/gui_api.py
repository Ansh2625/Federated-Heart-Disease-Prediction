import os, sys, json, math
from typing import Dict, Tuple, Callable
import numpy as np

# quiet TF + force CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTDIR = os.path.join(ROOT, "federated", "artifacts")
PLOTS  = os.path.join(ARTDIR, "plots")
GDATAD = os.path.join(ROOT, "federated", "data", "global")

# ---------- UI schema / sample ----------
def get_schema():
    return [
        {"name": "male", "label": "Male (0/1)", "type": "bool"},
        {"name": "age", "label": "Age (years)", "type": "int"},
        {"name": "education", "label": "Education (1-4)", "type": "int"},
        {"name": "currentSmoker", "label": "Current Smoker (0/1)", "type": "bool"},
        {"name": "cigsPerDay", "label": "Cigs per Day", "type": "float"},
        {"name": "BPMeds", "label": "BP Meds (0/1)", "type": "bool"},
        {"name": "prevalentStroke", "label": "Prev. Stroke (0/1)", "type": "bool"},
        {"name": "prevalentHyp", "label": "Prev. Hypertension (0/1)", "type": "bool"},
        {"name": "diabetes", "label": "Diabetes (0/1)", "type": "bool"},
        {"name": "totChol", "label": "Total Cholesterol", "type": "float"},
        {"name": "sysBP", "label": "Systolic BP", "type": "float"},
        {"name": "diaBP", "label": "Diastolic BP", "type": "float"},
        {"name": "BMI", "label": "BMI", "type": "float"},
        {"name": "heartRate", "label": "Heart Rate", "type": "float"},
        {"name": "glucose", "label": "Glucose", "type": "float"},
    ]

def get_sample_input() -> Dict[str, float]:
    return {
        "male": 1, "age": 52, "education": 3, "currentSmoker": 1,
        "cigsPerDay": 10.0, "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 1,
        "diabetes": 0, "totChol": 230.0, "sysBP": 135.0, "diaBP": 85.0,
        "BMI": 27.0, "heartRate": 78.0, "glucose": 95.0
    }

# ---------- preprocessing ----------
def _load_feature_order() -> list:
    p = os.path.join(GDATAD, "feature_columns.json")
    if os.path.isfile(p):
        return json.load(open(p))
    # fallback to centralized order
    cp = os.path.join(ROOT, "centralized", "data", "processed", "feature_names.npy")
    if os.path.isfile(cp):
        return list(np.load(cp, allow_pickle=True).tolist())
    raise FileNotFoundError("Missing federated/data/global/feature_columns.json")

def _load_imputer_scaler():
    from joblib import load
    imp_p = os.path.join(GDATAD, "global_imputer.joblib")
    scl_p = os.path.join(GDATAD, "global_scaler.joblib")
    if not os.path.isfile(imp_p) or not os.path.isfile(scl_p):
        raise FileNotFoundError("Missing global_imputer.joblib/global_scaler.joblib in federated/data/global")
    return load(imp_p), load(scl_p)

def _winsor_bounds(feature_order):
    low_p = os.path.join(GDATAD, "winsor_low.json")
    high_p = os.path.join(GDATAD, "winsor_high.json")
    if not (os.path.isfile(low_p) and os.path.isfile(high_p)):
        return None, None
    low_d  = json.load(open(low_p))
    high_d = json.load(open(high_p))
    low  = np.array([low_d.get(f, -np.inf)  for f in feature_order], dtype=float).reshape(1, -1)
    high = np.array([high_d.get(f,  np.inf) for f in feature_order], dtype=float).reshape(1, -1)
    return low, high

def _derive_feature(name: str, base: Dict[str, float]) -> float | None:
    """Compute common engineered features if requested in feature_columns.json."""
    n = name.lower().replace(" ", "").replace("-", "_")
    get = lambda k: float(base.get(k, 0.0))
    # aliases
    sbp = get("sysBP"); dbp = get("diaBP"); age = get("age"); bmi = get("BMI")
    chol = get("totChol"); glc = get("glucose"); cigs = get("cigsPerDay")
    smoker = get("currentSmoker")

    if n in ("pulse_pressure", "pulsepressure", "pp", "sbp_minus_dbp"):
        return sbp - dbp
    if n in ("bp_ratio", "sbp_over_dbp", "sbp_dbp_ratio"):
        return sbp / (dbp if dbp != 0 else 1e-6)
    if n in ("age_sq", "age2", "age^2"):
        return age * age
    if n in ("bmi_sq", "bmi2", "bmi^2"):
        return bmi * bmi
    if n in ("sysbp_sq", "sysbp2", "sysbp^2"):
        return sbp * sbp
    if n in ("diabp_sq", "diabp2", "diabp^2"):
        return dbp * dbp
    if n in ("log_glucose", "lnglucose", "log1p_glucose"):
        return float(np.log1p(glc))
    if n in ("log_totchol", "lntotchol", "log1p_totchol", "log_chol", "lnchol"):
        return float(np.log1p(chol))
    if n in ("cigs_sq", "cigsperday_sq", "cigs2", "cigs^2"):
        return cigs * cigs
    if n in ("smoker_x_cigs", "currentSmoker*cigsPerDay", "smoker_cigs"):
        return smoker * cigs
    # if it looks like a centered/standardized variant, fall back to raw (scaler will standardize)
    return None

def _vectorize(payload: Dict[str, float], feature_order: list) -> np.ndarray:
    # Use GUI base features when names match; otherwise try to derive
    row = []
    for k in feature_order:
        if k in payload:
            row.append(float(payload[k]))
        else:
            v = _derive_feature(k, payload)
            row.append(0.0 if v is None else float(v))
    return np.array(row, dtype=float).reshape(1, -1)

def _apply_preprocess(x: np.ndarray, feature_order: list) -> np.ndarray:
    imputer, scaler = _load_imputer_scaler()
    x_imp = imputer.transform(x)
    low, high = _winsor_bounds(feature_order)
    if low is not None and high is not None:
        x_imp = np.minimum(np.maximum(x_imp, low), high)  # clip like training
    x_scl = scaler.transform(x_imp)
    return x_scl

# ---------- model loading ----------
def _try_load_full_keras() -> Tuple[object, str] | None:
    import tensorflow as tf
    # 1) prefer fedavg.best.h5
    best = os.path.join(ARTDIR, "fedavg.best.h5")
    if os.path.isfile(best):
        try:
            m = tf.keras.models.load_model(best, compile=False)
            return m, "fedavg.best.h5"
        except Exception:
            pass
    # 2) any other .keras/.h5
    for f in sorted(os.listdir(ARTDIR), key=lambda n: n.lower()):
        if f.lower().endswith((".keras", ".h5")):
            try:
                m = tf.keras.models.load_model(os.path.join(ARTDIR, f), compile=False)
                return m, f
            except Exception:
                continue
    return None

def _build_from_weights(input_dim: int) -> Tuple[object, str] | None:
    ckpt_base = os.path.join(ARTDIR, "fedavg.weights")
    ckpt_pair = (os.path.isfile(ckpt_base + ".index") and
                 any(n.startswith("fedavg.weights.data-") for n in os.listdir(ARTDIR)))
    h5_weights = os.path.join(ARTDIR, "fedavg.weights.h5")
    best_h5    = os.path.join(ARTDIR, "fedavg.best.h5")

    sys.path.append(os.path.join(ROOT, "federated", "src"))
    import importlib, inspect, tensorflow as tf
    mdl = importlib.import_module("model")

    # collect candidate builders
    names = [
        "build_model","create_model","get_model","make_model","build_fed_model",
        "get_federated_model","fed_model","model_builder"
    ]
    fns = [getattr(mdl, n) for n in names if hasattr(mdl, n) and callable(getattr(mdl, n))]
    for name, obj in inspect.getmembers(mdl):
        if callable(obj) and obj not in fns:
            fns.append(obj)

    for fn in fns:
        for kwargs in ({"input_dim": input_dim}, {"input_shape": (input_dim,)}, {}):
            try:
                net = fn(**kwargs) if kwargs else fn()
                if not isinstance(net, tf.keras.Model):
                    continue
                # ensure variables exist
                if hasattr(net, "build"):
                    net.build((None, input_dim))
                try:
                    _ = net(np.zeros((1, input_dim), dtype=np.float32), training=False)
                except Exception:
                    pass
                # try weights
                if ckpt_pair:
                    try:
                        net.load_weights(ckpt_base)
                        return net, "fedavg.weights (ckpt)"
                    except Exception:
                        pass
                if os.path.isfile(h5_weights):
                    try:
                        net.load_weights(h5_weights)
                        return net, "fedavg.weights.h5"
                    except Exception:
                        pass
                if os.path.isfile(best_h5):
                    try:
                        net.load_weights(best_h5)
                        return net, "fedavg.best.h5 (weights)"
                    except Exception:
                        pass
            except Exception:
                continue
    return None

def _load_predictor(input_dim:int) -> Tuple[Callable[[np.ndarray], float], str]:
    fm = _try_load_full_keras()
    if fm:
        model, name = fm
        def _pred(X):
            y = model.predict(X, verbose=0)
            val = float(y[0] if hasattr(y, "__len__") else y)
            if val < 0 or val > 1:  # treat as logit if needed
                val = 1/(1+math.exp(-val))
            return val
        return _pred, name

    bw = _build_from_weights(input_dim)
    if bw:
        net, name = bw
        def _pred(X):
            y = net.predict(X, verbose=0)
            val = float(y[0] if hasattr(y, "__len__") else y)
            if val < 0 or val > 1:
                val = 1/(1+math.exp(-val))
            return val
        return _pred, name

    raise FileNotFoundError("No valid federated model (.keras/.h5) and no usable weights to rebuild.")

# ---------- public API ----------
def predict(payload: Dict, log=print) -> Dict:
    feats = _load_feature_order()
    x = _vectorize(payload, feats)
    x = _apply_preprocess(x, feats)
    pred_fn, model_name = _load_predictor(input_dim=x.shape[1])
    proba = float(np.clip(pred_fn(x), 0.0, 1.0))

    # optional custom threshold
    label = int(proba >= 0.5)
    thr_file = os.path.join(ARTDIR, "active_threshold.json")
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
