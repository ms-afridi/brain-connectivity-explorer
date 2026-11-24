from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectivity import global_efficiency


@dataclass
class StreamlineIndex:
    """Streamlines converted to voxel space once, so every lesion query is fast."""

    vox: np.ndarray          # (N, 3) int16, voxel index of every point
    lengths: np.ndarray      # (S,) points per streamline
    offsets: np.ndarray      # (S,) index of each streamline's first point
    end_idx: np.ndarray      # (S, 2) atlas region index (1..R) at both ends, 0 = none
    n_regions: int
    shape: tuple

    @property
    def n_streamlines(self) -> int:
        return len(self.lengths)


def build_index(points, lengths, affine, labels, region_ids) -> StreamlineIndex:
    """Convert RAS+ mm points to voxel indices and look up each streamline's end regions."""
    inv = np.linalg.inv(affine)
    vox_f = points.astype(np.float64) @ inv[:3, :3].T + inv[:3, 3]
    vox = np.rint(vox_f).astype(np.int16)           # same rounding DIPY uses

    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    ends = np.stack([offsets, offsets + lengths - 1], axis=1)   # first / last point

    lut = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    lut[np.asarray(region_ids)] = np.arange(1, len(region_ids) + 1)

    end_idx = np.zeros((len(lengths), 2), dtype=np.int64)
    for k in range(2):
        v = vox[ends[:, k]].astype(np.int64)
        ok = np.all((v >= 0) & (v < np.array(labels.shape)), axis=1)
        lab = np.zeros(len(v), dtype=np.int64)
        lab[ok] = labels[v[ok, 0], v[ok, 1], v[ok, 2]]
        end_idx[:, k] = lut[lab]

    return StreamlineIndex(vox, lengths, offsets, end_idx, len(region_ids), labels.shape)


def connectivity_matrix(index: StreamlineIndex, keep: np.ndarray | None = None) -> np.ndarray:
    """Region x region streamline counts (symmetric), using the end points only.

    Streamlines whose end points are not both inside an atlas region are ignored.
    """
    e = index.end_idx if keep is None else index.end_idx[keep]
    ok = (e[:, 0] > 0) & (e[:, 1] > 0)
    e = e[ok] - 1
    lo, hi = e.min(axis=1), e.max(axis=1)
    M = np.zeros((index.n_regions, index.n_regions), dtype=float)
    np.add.at(M, (lo, hi), 1)
    return np.maximum(M, M.T)


def sphere_mask(shape, center, radius) -> np.ndarray:
    """Boolean volume, True inside a sphere (centre and radius in voxels)."""
    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    d2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    return d2 <= radius ** 2


def streamlines_hit(index: StreamlineIndex, center, radius) -> np.ndarray:
    """Boolean per streamline: does it pass through the lesion sphere?"""
    c = np.asarray(center, dtype=np.int32)
    v = index.vox
    # cheap bounding-box test first, exact sphere test only on the points that survive it
    box = np.all(np.abs(v.astype(np.int32) - c) <= radius, axis=1)
    inside = np.zeros(len(v), dtype=bool)
    if box.any():
        d = v[box].astype(np.int32) - c
        inside[box] = (d * d).sum(axis=1) <= radius ** 2
    return np.add.reduceat(inside.astype(np.int32), index.offsets) > 0


def lesion_effect(index: StreamlineIndex, C_before: np.ndarray, center, radius,
                  min_streamlines: int = 3) -> dict:
    """Disconnection caused by one spherical lesion."""
    hit = streamlines_hit(index, center, radius)
    n_cut = int(hit.sum())
    C_after = connectivity_matrix(index, keep=~hit)

    total_before = C_before.sum()
    edges_before = C_before > 0

    return dict(
        radius=radius,
        n_cut=n_cut,
        streamlines_cut_pct=100 * n_cut / index.n_streamlines,
        connection_loss_pct=100 * (1 - C_after.sum() / total_before) if total_before else 0.0,
        edges_lost_pct=100 * ((edges_before & (C_after == 0)).sum() / edges_before.sum())
        if edges_before.sum() else 0.0,
        efficiency_before=global_efficiency(C_before, min_streamlines),
        efficiency_after=global_efficiency(C_after, min_streamlines),
        strength_drop=C_before.sum(axis=1) - C_after.sum(axis=1),
        C_after=C_after,
    )
