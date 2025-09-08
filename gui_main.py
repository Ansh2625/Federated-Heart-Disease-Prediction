# gui_main.py
import os, sys, threading, tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.abspath(__file__))

# make both adapters importable
sys.path.append(os.path.join(ROOT, "centralized", "src"))
sys.path.append(os.path.join(ROOT, "federated", "src"))

# import with distinct names (both are gui_api.py)
import importlib.util
def _load_adapter(path, modname):
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        return None

centralized_api = _load_adapter(os.path.join(ROOT, "centralized", "src", "gui_api.py"), "centralized_api")
federated_api   = _load_adapter(os.path.join(ROOT, "federated",   "src", "gui_api.py"), "federated_api")

APP_TITLE = "Heart Disease Risk — Centralized & Federated"
PAD = 8

def run_thread(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t

def _cast_num(s):
    s = (s or "").strip()
    if s == "": return None
    try:
        if "." in s: return float(s)
        return int(s)
    except: return None

class LogBox(ttk.Frame):
    def __init__(self, parent, h=10):
        super().__init__(parent)
        self.text = tk.Text(self, height=h, wrap="word", state="disabled")
        sb = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    def write(self, msg):
        self.text.configure(state="normal")
        self.text.insert("end", msg + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x680")
        try:
            ttk.Style(self).theme_use("clam")
        except: pass

        self.api_map = {"Centralized": centralized_api, "Federated": federated_api}
        self.current_api = None
        self.form_widgets = {}

        # topbar
        bar = ttk.Frame(self, padding=PAD)
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Approach:", font=("Segoe UI", 11, "bold")).pack(side="left")
        self.approach = tk.StringVar(value="Centralized")
        cb = ttk.Combobox(bar, textvariable=self.approach, values=["Centralized", "Federated"], state="readonly", width=16)
        cb.pack(side="left", padx=(PAD, PAD))
        cb.bind("<<ComboboxSelected>>", lambda e: self._switch(self.approach.get()))

        # main area: form (left) + result/plots (right)
        body = ttk.Frame(self, padding=PAD)
        body.pack(fill="both", expand=True)

        self.left = ttk.Frame(body)
        self.left.pack(side="left", fill="y", padx=(0, PAD))

        self.right = ttk.Notebook(body)
        self.right.pack(side="left", fill="both", expand=True)

        # Predict tab
        self.tab_predict = ttk.Frame(self.right, padding=PAD)
        self.right.add(self.tab_predict, text="Predict")

        # Plots tab
        self.tab_plots = ttk.Frame(self.right, padding=PAD)
        self.right.add(self.tab_plots, text="Visualizations")

        # Predict area
        self.form_group = ttk.LabelFrame(self.left, text="Enter Details (Framingham features)")
        self.form_group.pack(fill="y")

        btns = ttk.Frame(self.left)
        btns.pack(fill="x", pady=(PAD, 0))
        self.btn_sample  = ttk.Button(btns, text="Load Sample", command=self._load_sample)
        self.btn_predict = ttk.Button(btns, text="Predict", command=self._predict)
        self.btn_sample.pack(side="left", fill="x", expand=True, padx=(0, PAD))
        self.btn_predict.pack(side="left", fill="x", expand=True)

        # results
        res = ttk.LabelFrame(self.tab_predict, text="Prediction")
        res.pack(fill="both", expand=True)
        self.pred_label = ttk.Label(res, font=("Segoe UI", 18, "bold"))
        self.pred_label.pack(pady=(16, 4))
        self.pred_sub   = ttk.Label(res, font=("Segoe UI", 11))
        self.pred_sub.pack()
        self.log = LogBox(res, h=14)
        self.log.pack(fill="both", expand=True, pady=PAD)

        # plots
        self.plots_wrap = ttk.Frame(self.tab_plots)
        self.plots_wrap.pack(fill="both", expand=True)
        self.btn_refresh = ttk.Button(self.tab_plots, text="Refresh", command=self._load_plots)
        self.btn_refresh.pack(anchor="w")

        self._switch("Centralized")

    # --- Approach switch & form build ---
    def _switch(self, name):
        api = self.api_map.get(name)
        if api is None:
            messagebox.showerror("Missing adapter",
                                 f"{name} adapter failed to import. Check {name.lower()}/src/gui_api.py")
            return
        self.current_api = api
        for w in self.form_group.winfo_children(): w.destroy()
        self.form_widgets.clear()

        schema = self.current_api.get_schema()
        r, c = 0, 0
        for f in schema:
            cell = ttk.Frame(self.form_group)
            cell.grid(row=r, column=c, padx=PAD, pady=(4, 0), sticky="ew")
            ttk.Label(cell, text=f["label"]).pack(anchor="w")

            if f.get("type") == "bool":
                var = tk.StringVar(value="0")
                ent = ttk.Combobox(cell, state="readonly", values=["0", "1"], textvariable=var, width=12)
                ent.pack(fill="x")
                self.form_widgets[f["name"]] = var
            else:
                ent = ttk.Entry(cell)
                ent.pack(fill="x")
                self.form_widgets[f["name"]] = ent

            c += 1
            if c == 2:
                c = 0; r += 1

        self.form_group.grid_columnconfigure(0, weight=1)
        self.form_group.grid_columnconfigure(1, weight=1)

        self.pred_label.config(text="")
        self.pred_sub.config(text="")
        self.log.write(f"Switched to {name}.")
        self._load_plots()

    # --- helpers ---
    def _collect(self):
        data = {}
        schema = self.current_api.get_schema()
        types = {f["name"]: f.get("type") for f in schema}
        for name, widget in self.form_widgets.items():
            val = widget.get() if hasattr(widget, "get") else ""
            if types.get(name) == "bool":
                data[name] = 1 if str(val).strip() in ("1","true","True") else 0
            else:
                data[name] = _cast_num(val)
        return data

    # --- actions ---
    def _load_sample(self):
        if not self.current_api: return
        sample = self.current_api.get_sample_input()
        for k, v in sample.items():
            w = self.form_widgets.get(k)
            if w is None: continue
            if isinstance(w, tk.StringVar):
                w.set(str(v))
            else:
                w.delete(0, "end")
                w.insert(0, str(v))
        self.log.write("Loaded sample input.")

    def _predict(self):
        if not self.current_api: return
        payload = self._collect()
        self.pred_label.config(text="Predicting...")
        self.pred_sub.config(text="")
        self.log.write("Preparing inference...")

        def job():
            try:
                result = self.current_api.predict(payload, log=self.log.write)
                self.pred_label.config(text=result.get("proba_text","N/A"))
                self.pred_sub.config(text=result.get("extra",""))
            except Exception as e:
                self.pred_label.config(text="Prediction failed")
                self.pred_sub.config(text="")
                self.log.write(f"[ERROR] {type(e).__name__}: {e}")

        run_thread(job)

    def _load_plots(self):
        for w in self.plots_wrap.winfo_children(): w.destroy()
        if not self.current_api: return
        plot_dir = self.current_api.get_plots_dir()

        imgs = []
        if os.path.isdir(plot_dir):
            for f in os.listdir(plot_dir):
                if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".gif")):
                    imgs.append(os.path.join(plot_dir,f))
        if not imgs:
            ttk.Label(self.plots_wrap, text=f"No plot images found in: {plot_dir}").pack(padx=PAD, pady=PAD)
            return

        canvas = tk.Canvas(self.plots_wrap)
        vsb = ttk.Scrollbar(self.plots_wrap, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._img_refs = []
        for p in sorted(imgs):
            try:
                im = Image.open(p)
                im.thumbnail((900, 700))
                ph = ImageTk.PhotoImage(im)
                self._img_refs.append(ph)
                ttk.Label(inner, image=ph, text=os.path.basename(p), compound="top").pack(padx=PAD, pady=PAD)
            except Exception as e:
                ttk.Label(inner, text=f"Failed to load {os.path.basename(p)}: {e}").pack()

if __name__ == "__main__":
    App().mainloop()
