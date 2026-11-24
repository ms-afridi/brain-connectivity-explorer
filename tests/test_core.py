import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.connectivity import graph_metrics
from src.lesions import build_index, connectivity_matrix, lesion_effect
from src.loaders import MODELS, load_streamlines, load_volumes
from src.prediction import OasisModel


@pytest.fixture(scope="module")
def data():
    v = load_volumes()
    pts, lens = load_streamlines()
    idx = build_index(pts, lens, v["affine"], v["labels"], v["region_ids"])
    return v, idx, connectivity_matrix(idx)


def test_matrix_is_symmetric_and_nonempty(data):
    _, _, C = data
    assert C.shape == (86, 86)
    assert np.allclose(C, C.T)
    assert C.sum() > 0


def test_tiny_lesion_outside_brain_changes_nothing(data):
    _, idx, C = data
    res = lesion_effect(idx, C, center=(0, 0, 0), radius=1)
    assert res["n_cut"] == 0
    assert res["connection_loss_pct"] == 0
    assert np.array_equal(res["C_after"], C)


def test_bigger_lesion_cuts_more(data):
    _, idx, C = data
    cut = [lesion_effect(idx, C, (39, 42, 38), r)["streamlines_cut_pct"] for r in (2, 6, 10)]
    assert cut[0] < cut[1] < cut[2]


def test_graph_metrics_in_range(data):
    _, _, C = data
    gm = graph_metrics(C)
    assert 0 < gm["global_efficiency"] < 1
    assert 0 < gm["density"] < 1


def test_prediction_backends_agree():
    m = OasisModel()
    person = {f: m.feature_ranges()[f][0] for f in m.features}
    p = m.predict_proba(person)
    assert 0 <= p <= 1
    if m._pipe is not None:  # compare scikit-learn with the numpy fallback
        z = (m._vector(person) - m.mean) / m.std
        p_np = 1 / (1 + np.exp(-(m.intercept + z @ m.coef)))
        assert abs(p - p_np) < 1e-6


def test_matches_dipy(data):
    pytest.importorskip("dipy")
    from dipy.io.streamline import load_trk
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines

    v, idx, C = data
    sft = load_trk(str(MODELS / "streamlines_small.trk"), "same", bbox_valid_check=False)
    sft.to_rasmm()
    sl = Streamlines(sft.streamlines)
    labels = v["labels"].astype(int)
    M = utils.connectivity_matrix(sl, v["affine"], labels, symmetric=True)
    M[0, :] = 0
    M[:, 0] = 0
    Cd = M[np.ix_(v["region_ids"], v["region_ids"])].astype(float)
    assert np.array_equal(C, Cd)
