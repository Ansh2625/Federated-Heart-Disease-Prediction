# src/common/paths.py
import os

class Paths:
    def __init__(self):
        # repos that run scripts inside src/... need root = two dirs up
        self.SRC = os.path.dirname(os.path.abspath(__file__))           # .../src/common
        self.ROOT = os.path.dirname(self.SRC)                            # .../src
        self.PROJ = os.path.dirname(self.ROOT)                           # repo root

        self.DATA_RAW = os.path.join(self.PROJ, "data", "raw")
        self.DATA_PROCESSED = os.path.join(self.PROJ, "data", "processed")

        self.ARTIFACTS_CENT = os.path.join(self.PROJ, "artifacts", "centralized")
        self.PLOTS_CENT = os.path.join(self.ARTIFACTS_CENT, "plots")
        self.ARTIFACTS_FED = os.path.join(self.PROJ, "artifacts", "federated")

        # ensure dirs exist
        for d in [self.DATA_RAW, self.DATA_PROCESSED,
                  self.ARTIFACTS_CENT, self.PLOTS_CENT, self.ARTIFACTS_FED]:
            os.makedirs(d, exist_ok=True)
