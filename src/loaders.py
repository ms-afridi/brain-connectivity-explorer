from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def load_volumes(path: Path | None = None) -> dict:
    """FA, MD, atlas labels, brain mask, connectivity matrix (40k streamlines), affine."""
    z = np.load(path or MODELS / "volumes.npz")
    return {k: z[k] for k in z.files}


def load_streamlines(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (points, lengths): all streamline points stacked (N, 3) in RAS+ mm,
    and the number of points of each streamline.

    The .trk file was saved by DIPY in RAS+ mm space; nibabel returns the same space.
    """
    tr = nib.streamlines.load(str(path or MODELS / "streamlines_small.trk"))
    sls = tr.streamlines
    lengths = np.array([len(s) for s in sls], dtype=np.int64)
    points = np.concatenate([np.asarray(s, dtype=np.float32) for s in sls], axis=0)
    return points, lengths


def load_meta(path: Path | None = None) -> dict:
    with open(path or MODELS / "oasis_meta.json") as f:
        return json.load(f)


def load_lesion_sweep(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or MODELS / "lesion_sweep.csv")
