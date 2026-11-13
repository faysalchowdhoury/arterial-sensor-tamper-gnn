"""PEMS-BAY loop-detector speeds and the sensor graph.

PEMS-BAY: 325 Caltrans PeMS sensors in the San Francisco Bay Area, 5-min average speed (mph),
1 Jan - 30 Jun 2017 (Li et al., DCRNN, 2018; CSV mirror on Zenodo, record 5146275).

The graph is built from the sensor metadata, not from the distributed ``adj_mx_bay.pkl``: that file is a
Python pickle, and unpickling a downloaded file can execute arbitrary code. Edges:

* along a freeway: each sensor to its two nearest neighbours upstream and downstream on the same
  freeway and direction (by absolute postmile);
* across freeways: sensors within 1.0 km of each other (interchanges, parallel routes).

Weights are a Gaussian kernel of distance, :math:`w = \\exp(-(d/\\sigma)^2)`, with sigma the standard
deviation of edge lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

MI = 1.609344


@dataclass(frozen=True)
class Dataset:
    speed: np.ndarray  # (T, N) mph
    times: pd.DatetimeIndex
    sensors: np.ndarray  # (N,) sensor ids
    meta: pd.DataFrame  # aligned with sensors
    edge_index: np.ndarray  # (2, E), both directions
    edge_weight: np.ndarray  # (E,)


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build_graph(meta: pd.DataFrame, *, k_along: int = 2, cross_km: float = 1.0) -> tuple[np.ndarray, np.ndarray, nx.Graph]:
    g = nx.Graph()
    g.add_nodes_from(range(len(meta)))
    for _, grp in meta.groupby(["Fwy", "Dir"]):
        idx = grp.sort_values("Abs_PM").index.to_numpy()
        pm = meta.loc[idx, "Abs_PM"].to_numpy()
        for a in range(len(idx)):
            for b in range(a + 1, min(a + 1 + k_along, len(idx))):
                g.add_edge(int(idx[a]), int(idx[b]), d=abs(pm[b] - pm[a]) * MI)
    lat, lon = meta["Latitude"].to_numpy(), meta["Longitude"].to_numpy()
    d = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    ii, jj = np.where((d < cross_km) & (np.triu(np.ones_like(d), 1) > 0))
    for i, j in zip(ii, jj):
        if not g.has_edge(int(i), int(j)):
            g.add_edge(int(i), int(j), d=float(d[i, j]))
    dist = np.array([max(e["d"], 0.05) for _, _, e in g.edges(data=True)])
    sigma = dist.std() + 1e-6
    src, dst, w = [], [], []
    for (u, v, e), dd in zip(g.edges(data=True), dist):
        ww = float(np.exp(-((dd / sigma) ** 2)))
        src += [u, v]
        dst += [v, u]
        w += [ww, ww]
    return np.array([src, dst]), np.array(w), g


def load(data_dir: str | Path) -> Dataset:
    p = Path(data_dir)
    df = pd.read_csv(p / "PEMS-BAY.csv", index_col=0, parse_dates=True)
    meta = pd.read_csv(p / "PEMS-BAY-META.csv")
    sensors = df.columns.astype(int).to_numpy()
    meta = meta.set_index("sensor_id").loc[sensors].reset_index()
    speed = df.to_numpy(dtype=np.float32)
    # PeMS marks missing samples as 0: fill from the previous sample of the same sensor
    s = pd.DataFrame(np.where(speed <= 0, np.nan, speed)).ffill().bfill().to_numpy(np.float32)
    ei, ew, _ = build_graph(meta)
    return Dataset(s, df.index, sensors, meta, ei, ew)


def split(times: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Train Jan-Mar, validation Apr, test May-Jun (index masks)."""
    m = times.month
    return {"train": np.flatnonzero(m <= 3), "val": np.flatnonzero(m == 4), "test": np.flatnonzero(m >= 5)}
