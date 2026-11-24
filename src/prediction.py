from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from .loaders import MODELS, load_meta

FEATURE_LABELS = {
    "sex": "Sex",
    "Age": "Age",
    "EDUC": "Education",
    "SES": "Socioeconomic status",
    "eTIV": "Head size (eTIV)",
    "nWBV": "Brain volume (nWBV)",
}


class OasisModel:
    def __init__(self):
        self.meta = load_meta()
        self.features = self.meta["features"]
        with open(MODELS / "oasis_params.json") as f:
            p = json.load(f)
        self.median = np.array(p["impute_median"])
        self.mean = np.array(p["scale_mean"])
        self.std = np.array(p["scale_std"])
        self.coef = np.array(p["coef"])
        self.intercept = float(p["intercept"])
        self.backend = "numpy (oasis_params.json)"
        self._pipe = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._pipe = joblib.load(MODELS / "oasis_model.pkl")
            self.backend = "scikit-learn (oasis_model.pkl)"
        except Exception:
            pass  # fall back to the numpy version

    def _vector(self, values: dict) -> np.ndarray:
        return np.array([values[f] for f in self.features], dtype=float)

    def predict_proba(self, values: dict) -> float:
        """Probability of the 'Demented' class for one person."""
        x = self._vector(values)
        if self._pipe is not None:
            X = pd.DataFrame([x], columns=self.features)  # the pipeline was fitted on named columns
            return float(self._pipe.predict_proba(X)[0, 1])
        z = (x - self.mean) / self.std
        return float(1 / (1 + np.exp(-(self.intercept + z @ self.coef))))

    def contributions(self, values: dict) -> dict:
        """How much each input moves the log-odds away from the 'average person'.
        Positive pushes toward Demented, negative toward Nondemented."""
        z = (self._vector(values) - self.mean) / self.std
        return dict(zip(self.features, (z * self.coef).tolist()))

    def feature_ranges(self) -> dict:
        """(default, low, high) for each input, from the training data statistics."""
        out = {}
        for f, med, m, s in zip(self.features, self.median, self.mean, self.std):
            out[f] = (float(med), float(m - 3 * s), float(m + 3 * s))
        return out
