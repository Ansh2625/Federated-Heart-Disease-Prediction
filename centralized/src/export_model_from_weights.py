# centralized/src/export_model_from_weights.py
import os, sys, inspect
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART  = os.path.join(ROOT, "centralized", "artifacts")
DATA = os.path.join(ROOT, "centralized", "data", "processed")

WEIGHTS = os.path.join(ART, "best_model.weights.h5")
OUTPATH = os.path.join(ART, "model.keras")  # GUI will load this

def get_input_dim() -> int:
    names = os.path.join(DATA, "feature_names.npy")
    if not os.path.isfile(names):
        raise FileNotFoundError("centralized/data/processed/feature_names.npy not found")
    return int(len(np.load(names, allow_pickle=True)))

def find_and_build_model(input_dim: int):
    sys.path.append(os.path.join(ROOT, "centralized", "src"))
    import importlib, tensorflow as tf

    m = importlib.import_module("model")  # your centralized/src/model.py

    # candidate builder names + any callable returning a Keras Model
    candidates = [
        "build_model","create_model","get_model","make_model","build_baseline_model",
        "get_centralized_model","baseline_model","model_builder"
    ]
    fns = []
    for name in candidates:
        if hasattr(m, name) and callable(getattr(m, name)):
            fns.append(getattr(m, name))
    for name, obj in inspect.getmembers(m):
        if callable(obj) and obj not in fns:
            fns.append(obj)

    last_err = None
    for fn in fns:
        for kwargs in ({"input_dim": input_dim}, {"input_shape": (input_dim,)}, {}):
            try:
                mdl = fn(**kwargs) if kwargs else fn()
                # ensure it's a Keras Model
                if not isinstance(mdl, tf.keras.Model):
                    continue
                # *** BUILD MODEL BEFORE LOADING WEIGHTS ***
                if hasattr(mdl, "build"):
                    mdl.build((None, input_dim))
                try:
                    # dummy forward pass to create variables for subclassed models
                    _ = mdl(np.zeros((1, input_dim), dtype=np.float32), training=False)
                except Exception:
                    pass
                return mdl
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Could not build model from centralized/src/model.py. Last error: {last_err}")

def main():
    import tensorflow as tf
    if not os.path.isfile(WEIGHTS):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

    d = get_input_dim()
    model = find_and_build_model(d)

    # load weights AFTER building & dummy pass
    model.load_weights(WEIGHTS)

    # Save a full model for the GUI to load.
    # Keras 3: DO NOT pass include_optimizer when saving `.keras`
    try:
        model.save(OUTPATH)  # saves native Keras format
        print(f"[OK] Exported full model to: {OUTPATH}")
    except Exception as e:
        # fallback to legacy H5 if needed
        out_h5 = os.path.join(ART, "model.h5")
        model.save(out_h5)   # legacy H5 format
        print(f"[OK] Exported full model to: {out_h5} (legacy H5) due to: {e}")

if __name__ == "__main__":
    main()
