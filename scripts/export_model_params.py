import json
import warnings
from pathlib import Path

import joblib

warnings.simplefilter("ignore")
models = Path(__file__).resolve().parents[1] / "models"
pipe = joblib.load(models / "oasis_model.pkl")
clf = pipe.named_steps["clf"]
params = {
    "impute_median": pipe.named_steps["impute"].statistics_.tolist(),
    "scale_mean": pipe.named_steps["scale"].mean_.tolist(),
    "scale_std": pipe.named_steps["scale"].scale_.tolist(),
    "coef": clf.coef_[0].tolist(),
    "intercept": float(clf.intercept_[0]),
}
(models / "oasis_params.json").write_text(json.dumps(params, indent=2))
print("wrote", models / "oasis_params.json")
