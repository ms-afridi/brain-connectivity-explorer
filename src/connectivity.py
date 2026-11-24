from __future__ import annotations

import networkx as nx
import numpy as np


def _graph(C: np.ndarray, min_streamlines: int = 3) -> nx.Graph:
    A = (C >= min_streamlines).astype(int)
    np.fill_diagonal(A, 0)
    return nx.from_numpy_array(A)


def global_efficiency(C: np.ndarray, min_streamlines: int = 3) -> float:
    return float(nx.global_efficiency(_graph(C, min_streamlines)))


def graph_metrics(C: np.ndarray, min_streamlines: int = 3) -> dict:
    """Same measures as the notebook. A region pair counts as connected when it has
    at least `min_streamlines` streamlines (weak 1-2 streamline links are mostly noise)."""
    G = _graph(C, min_streamlines)
    return {
        "node_strength": C.sum(axis=1),
        "global_efficiency": float(nx.global_efficiency(G)),
        "density": float(nx.density(G)),
    }
