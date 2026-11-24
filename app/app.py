import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter

from src.connectivity import graph_metrics
from src.lesions import build_index, connectivity_matrix, lesion_effect, sphere_mask
from src.loaders import load_lesion_sweep, load_streamlines, load_volumes
from src.prediction import FEATURE_LABELS, OasisModel

st.set_page_config(page_title="Brain Connectivity Explorer", layout="wide")

LESION_COLOR = "#D1495B"
BLUE = "#2F5D8A"


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading brain volumes...")
def get_volumes():
    return load_volumes()


@st.cache_resource(show_spinner="Loading streamlines (first run only)...")
def get_index():
    v = get_volumes()
    points, lengths = load_streamlines()
    idx = build_index(points, lengths, v["affine"], v["labels"], v["region_ids"])
    return idx, connectivity_matrix(idx)


@st.cache_resource
def get_model():
    return OasisModel()


@st.cache_data
def default_lesion_center():
    """Centre of the densest white matter: highest smoothed FA (same rule as the notebook)."""
    v = get_volumes()
    smooth = gaussian_filter(v["fa"], sigma=2) * v["mask"]
    return tuple(int(i) for i in np.unravel_index(np.argmax(smooth), v["fa"].shape))


vol = get_volumes()
fa, md, labels, mask = vol["fa"], vol["md"], vol["labels"], vol["mask"]
shape = fa.shape
voxel_mm = float(abs(vol["affine"][0, 0]))
PLANES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}


def take_slice(arr, plane, k):
    axis = PLANES[plane]
    return np.take(arr, k, axis=axis).T  # transpose so the brain is upright


def slice_figure(base, plane, k, base_scale="gray", zmax=1.0, overlay=None, overlay_kind=None,
                 title=None, height=430):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=take_slice(base, plane, k), colorscale=base_scale, zmin=0, zmax=zmax,
                             showscale=False, hoverinfo="skip"))
    if overlay is not None:
        o = take_slice(overlay, plane, k)
        if overlay_kind == "lesion":
            fig.add_trace(go.Heatmap(z=np.where(o, 1.0, np.nan), colorscale=[[0, LESION_COLOR], [1, LESION_COLOR]],
                                     opacity=0.6, showscale=False, hoverinfo="skip"))
        else:  # atlas regions
            fig.add_trace(go.Heatmap(z=np.where(o > 0, o, np.nan), colorscale="Turbo", opacity=0.45,
                                     showscale=False, hovertemplate="Region %{z}<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=0, r=0, t=30 if title else 0, b=0), height=height, title=title)
    return fig


# ----------------------------------------------------------------------------- header
st.title("Brain Connectivity Explorer")
st.caption(
    "A prototype that goes from diffusion MRI to a map of white matter connections, simulates damage to "
    "those connections, and (separately) predicts dementia status from demographics and brain volume. "
    "Imaging data: one healthy subject (Stanford HARDI). Tabular data: OASIS. Lesions are simulated."
)

tab_maps, tab_conn, tab_lesion, tab_model, tab_about = st.tabs(
    ["Brain maps", "Connectivity", "Lesion simulation", "Dementia model", "About and limits"]
)

# ----------------------------------------------------------------------------- tab 1
with tab_maps:
    c1, c2 = st.columns([1, 3])
    with c1:
        what = st.radio("Map", ["FA", "MD", "Atlas regions"], help=(
            "FA (fractional anisotropy): how directional water diffusion is. Bright = organised white matter. "
            "MD (mean diffusivity): how freely water moves. Bright = fluid, such as the ventricles."))
        plane = st.radio("Plane", list(PLANES), horizontal=True)
        n = shape[PLANES[plane]]
        k = st.slider("Slice", 0, n - 1, n // 2)
        show_atlas = st.checkbox("Overlay atlas regions", value=False, disabled=(what == "Atlas regions"))
    with c2:
        if what == "FA":
            fig = slice_figure(fa, plane, k, "gray", 1.0, labels if show_atlas else None, "atlas")
        elif what == "MD":
            zmax = float(np.percentile(md[mask], 99))
            fig = slice_figure(md, plane, k, "Viridis", zmax, labels if show_atlas else None, "atlas")
        else:
            fig = slice_figure(fa, plane, k, "gray", 1.0, labels, "atlas")
        st.plotly_chart(fig)
    st.caption("Source: Stanford HARDI (160 diffusion volumes, b = 2000 s/mm2), DTI fitted with DIPY. "
               "Voxels are %g mm wide." % voxel_mm)

# ----------------------------------------------------------------------------- tab 2
with tab_conn:
    C_nb = vol["C"]
    region_ids = vol["region_ids"]
    c1, c2 = st.columns([1, 3])
    with c1:
        thr = st.slider("Minimum streamlines for a connection", 1, 20, 3,
                        help="A region pair counts as connected only if at least this many streamlines "
                             "join them. 1-2 streamline links are mostly tractography noise.")
        gm = graph_metrics(C_nb, thr)
        st.metric("Global efficiency", "%.3f" % gm["global_efficiency"])
        st.metric("Network density", "%.3f" % gm["density"])
        st.metric("Connected region pairs", int(((C_nb >= thr).sum() - np.trace(C_nb >= thr)) // 2))
        st.caption("Matrix built from 40,000 streamlines between %d atlas regions." % len(region_ids))
    with c2:
        ids = [str(r) for r in region_ids]
        hm = go.Figure(go.Heatmap(z=np.log1p(C_nb), x=ids, y=ids, colorscale="Magma",
                                  colorbar=dict(title="log(1 + streamlines)"),
                                  hovertemplate="Region %{x} - Region %{y}<br>log(1+n) = %{z:.2f}<extra></extra>"))
        hm.update_xaxes(type="category", tickmode="array", tickvals=ids[::10], title="Region ID")
        hm.update_yaxes(type="category", tickmode="array", tickvals=ids[::10], autorange="reversed",
                        scaleanchor="x", title="Region ID")
        hm.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520)
        st.plotly_chart(hm)

    st.subheader("Best connected regions")
    strength = gm["node_strength"]
    top = np.argsort(strength)[::-1][:10]
    bar = go.Figure(go.Bar(x=[str(region_ids[i]) for i in top][::-1], y=strength[top][::-1],
                           marker_color=BLUE, orientation="v"))
    bar.update_xaxes(type="category", title="Region ID")
    bar.update_yaxes(title="Node strength (total streamlines)")
    bar.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(bar)
    st.caption("Region IDs are the numeric labels of the Stanford atlas (names were not exported from the notebook).")

# ----------------------------------------------------------------------------- tab 3
with tab_lesion:
    index, C_intact = get_index()
    dx, dy, dz = default_lesion_center()

    def reset_lesion():
        st.session_state.lx, st.session_state.ly, st.session_state.lz, st.session_state.lr = dx, dy, dz, 6

    for key, val in (("lx", dx), ("ly", dy), ("lz", dz), ("lr", 6)):
        st.session_state.setdefault(key, val)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("**Place a simulated lesion**")
        st.slider("Left to right (voxel)", 0, shape[0] - 1, key="lx")
        st.slider("Back to front (voxel)", 0, shape[1] - 1, key="ly")
        st.slider("Bottom to top (voxel)", 0, shape[2] - 1, key="lz")
        st.slider("Radius (voxels)", 1, 15, key="lr")
        st.caption("Radius = %g mm. Any streamline that touches the sphere is treated as cut." %
                   (st.session_state.lr * voxel_mm))
        st.button("Reset to the notebook's lesion", on_click=reset_lesion)

    center = (st.session_state.lx, st.session_state.ly, st.session_state.lz)
    radius = st.session_state.lr
    res = lesion_effect(index, C_intact, center, radius)
    lesion_vol = sphere_mask(shape, center, radius)

    with c2:
        a, b = st.columns(2)
        with a:
            st.plotly_chart(slice_figure(fa, "Axial", center[2], "gray", 1.0, lesion_vol, "lesion",
                                         title="Axial slice through the lesion", height=340))
        with b:
            st.plotly_chart(slice_figure(fa, "Coronal", center[1], "gray", 1.0, lesion_vol, "lesion",
                                         title="Coronal slice through the lesion", height=340))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Streamlines cut", "%.1f%%" % res["streamlines_cut_pct"],
              help="%d of %d streamlines pass through the lesion." % (res["n_cut"], index.n_streamlines))
    m2.metric("Connection strength lost", "%.1f%%" % res["connection_loss_pct"],
              help="Drop in the total streamline count of the connectivity matrix.")
    m3.metric("Region pairs fully disconnected", "%.1f%%" % res["edges_lost_pct"],
              help="Region pairs that had at least one streamline before and none after.")
    m4.metric("Global efficiency", "%.3f" % res["efficiency_after"],
              "%+.3f" % (res["efficiency_after"] - res["efficiency_before"]))

    if res["n_cut"] == 0:
        st.info("No fibres pass through this lesion. Move it into white matter (the bright areas on the FA "
                "image) or make it bigger.")
    else:
        before = C_intact.sum(axis=1)
        drop = res["strength_drop"]
        order = np.argsort(drop)[::-1][:8]
        table = pd.DataFrame({
            "Region ID": [int(vol["region_ids"][i]) for i in order],
            "Streamlines before": before[order].astype(int),
            "Streamlines after": (before[order] - drop[order]).astype(int),
            "Lost (%)": np.round(100 * drop[order] / np.maximum(before[order], 1), 1),
        })
        st.markdown("**Most affected regions**")
        st.dataframe(table, hide_index=True)

    st.divider()
    st.markdown("**Lesion size versus damage (from the notebook)**")
    sweep = load_lesion_sweep()
    sw = go.Figure()
    sw.add_trace(go.Scatter(x=sweep["radius"], y=sweep["streamlines_cut_pct"], mode="lines+markers",
                            name="Streamlines cut", line=dict(color=BLUE)))
    sw.add_trace(go.Scatter(x=sweep["radius"], y=sweep["edges_lost_pct"], mode="lines+markers",
                            name="Region pairs fully disconnected", line=dict(color=LESION_COLOR)))
    sw.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, xaxis_title="Lesion radius (voxels)",
                     yaxis_title="%", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(sw)
    st.caption("The live numbers above use a 15,000-streamline sample so the sliders stay fast. The notebook "
               "sweep used 40,000 streamlines, so values differ slightly. Lesions are synthetic spheres, not "
               "patient data.")

# ----------------------------------------------------------------------------- tab 4
with tab_model:
    model = get_model()
    meta = model.meta
    rng = model.feature_ranges()

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Enter a person's details**")
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True, index=int(rng["sex"][0] >= 0.5))
        age = st.slider("Age (years)", 50, 100, int(round(rng["Age"][0])))
        educ = st.slider("Education (years)", 0, 25, int(round(rng["EDUC"][0])))
        ses = st.select_slider("Socioeconomic status (1 = highest, 5 = lowest)", [1, 2, 3, 4, 5],
                               value=int(round(rng["SES"][0])))
        etiv = st.slider("Head size, eTIV (cm3)", 900, 2100, int(round(rng["eTIV"][0], -1)), step=10)
        nwbv = st.slider("Brain volume, nWBV (fraction of head size)", 0.60, 0.88,
                         float(round(rng["nWBV"][0], 3)), step=0.001, format="%.3f")
    person = {"sex": 1 if sex == "Male" else 0, "Age": age, "EDUC": educ, "SES": ses, "eTIV": etiv, "nWBV": nwbv}
    p = model.predict_proba(person)

    with right:
        st.metric("Estimated probability of the Demented group", "%.0f%%" % (100 * p))
        st.progress(min(max(p, 0.0), 1.0))
        st.caption("Research demo, not a diagnostic tool. The model uses class weighting, so this is a "
                   "relative score and not a real-world risk.")
        contrib = model.contributions(person)
        items = sorted(contrib.items(), key=lambda kv: kv[1])
        cf = go.Figure(go.Bar(
            x=[v for _, v in items], y=[FEATURE_LABELS[k] for k, _ in items], orientation="h",
            marker_color=[LESION_COLOR if v > 0 else BLUE for _, v in items]))
        cf.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300,
                         title="What pushes this estimate up or down",
                         xaxis_title="Effect on log-odds (positive = toward Demented)")
        st.plotly_chart(cf)
        st.caption("These are not causal effects. Age pushes the estimate down here, most likely because brain "
                   "volume already captures age-related shrinkage, so at a fixed brain volume an older person "
                   "looks relatively healthier. Read the chart as how this model weighs its inputs, not as "
                   "medical findings.")

    st.divider()
    st.markdown("**How good is the model?**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", "%.1f%%" % (100 * meta["accuracy"]), "± %.1f" % (100 * meta["accuracy_sd"]),
              delta_color="off")
    k2.metric("ROC-AUC", "%.2f" % meta["roc_auc"])
    k3.metric("Visits used", meta["n_rows"])
    k4.metric("Model", "Logistic regression")
    st.caption("Task: %s. Validation: %s. Inputs are demographics and brain volume only. MMSE and CDR are "
               "left out on purpose, because the dementia label is defined from them and using them would "
               "leak the answer. Loaded with: %s." % (meta["task"], meta["validation"], model.backend))

# ----------------------------------------------------------------------------- tab 5
with tab_about:
    st.markdown("""
**What this prototype does**

1. Fits a diffusion tensor to a diffusion MRI scan and maps white matter organisation (FA, MD).
2. Tracks fibres through the brain and counts how many join each pair of atlas regions, giving a
   structural connectivity matrix and simple network measures.
3. Places a simulated lesion and measures how many fibres and region-to-region connections it disrupts.
4. Separately, trains and serves a small model that predicts dementia status from OASIS demographics and
   brain volume.

**Data**

- Diffusion MRI and atlas: Stanford HARDI, downloaded through DIPY (one healthy adult).
- Tabular data: OASIS longitudinal, from Kaggle.
- Lesions: synthetic spheres generated by the code.

**Limits to keep in mind**

- The imaging comes from one healthy person and the tabular data from other people. The two parts are
  independent demonstrations. No analysis here links a person's connectivity to their cognition.
- Lesions are simulated spheres, not real vascular or neurodegenerative damage.
- The scan uses a single high b-value (2000 s/mm2), higher than typical DTI protocols, so FA and MD values
  differ from clinical norms.
- Deterministic tractography on one subject includes some false connections. Graph measures depend on the
  minimum-streamline threshold.
- OASIS has about 150 people. The model reaches roughly 70% accuracy with a spread of about 3 points, and
  subjects labelled Converted were excluded because their label is ambiguous.

**What a full study would change**

Use patient diffusion MRI together with cognitive scores from the same people, derive lesion masks and
connectivity per person, and test whether connection damage improves diagnosis, staging and prediction of
decline beyond what demographics and brain volume already give.
""")
