"""Predict each sensor's *current* speed without looking at its own current reading.

Tampering shows up as the gap between what a sensor reports and what the rest of the network says it
should report. Three predictors:

* ``persistence``: the sensor's own previous reading (no spatial information);
* ``spatial``: the previous reading scaled by how the sensor's neighbours changed since then,
  :math:`\\hat x_{i,t} = x_{i,t-1}\\,\\bar x_{\\mathcal N(i),t} / \\bar x_{\\mathcal N(i),t-1}`;
* ``stgnn``: a spatio-temporal graph network. A GRU encodes each sensor's last hour; each neighbour also
  contributes its *current* reading; a weighted one-hop aggregation of neighbour embeddings plus the
  sensor's own history embedding predicts a correction to its previous reading. The sensor's own current
  value is never an input, so the prediction cannot simply copy a tampered reading.
* ``stgnn_nb`` (neighbours only): the same network without the sensor's own history. A learned
  per-sensor embedding replaces it, so the sensor's level relative to its neighbours is learned once in
  training. Once tampering starts, a sensor's own *recent* readings are tampered too, and a predictor that
  leans on them follows the lie (a stuck sensor predicts itself perfectly). This variant cannot.

Training adds **neighbour corruption**: 5% of nodes in each batch get a random fake current reading as
*inputs to their neighbours*, so the model learns not to follow one odd neighbour blindly.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

HIST = 12  # one hour of 5-min samples
SCALE = 70.0


def neighbour_mean(x: np.ndarray, edge_index: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Weighted mean over neighbours for each row of ``x`` (T, N)."""
    src, dst = edge_index
    num = np.zeros((x.shape[0], n))
    den = np.zeros(n)
    np.add.at(num.T, dst, (x[:, src] * w).T)
    np.add.at(den, dst, w)
    return num / np.maximum(den, 1e-9)


def predict_persistence(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x[t - 1]


def predict_spatial(x: np.ndarray, t: np.ndarray, edge_index: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    now = neighbour_mean(x[t], edge_index, w, n)
    before = neighbour_mean(x[t - 1], edge_index, w, n)
    return np.clip(x[t - 1] * now / np.maximum(before, 1.0), 0, 90)


class STGNN(nn.Module):
    def __init__(self, hidden: int = 32, *, own_history: bool = True, n_nodes: int = 325) -> None:
        super().__init__()
        self.own_history = own_history
        self.node = nn.Embedding(n_nodes, hidden)
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.nb = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, hist: torch.Tensor, cur: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """hist (B, N, HIST), cur (B, N) neighbour-visible current readings -> prediction (B, N), scaled units."""
        b, n, h = hist.shape
        _, z = self.gru(hist.reshape(b * n, h, 1))
        z = z[-1].reshape(b, n, -1)
        msg_in = self.nb(torch.cat([z, cur.unsqueeze(-1)], -1))  # what each node tells its neighbours
        src, dst = edge_index
        agg = torch.zeros_like(msg_in)
        agg.index_add_(1, dst, msg_in[:, src] * w[None, :, None])
        den = torch.zeros(n).index_add_(0, dst, w).clamp_min(1e-9)
        agg = agg / den[None, :, None]
        if not self.own_history:  # neighbours only: the sensor's own values never enter its prediction
            nb_now = torch.zeros(b, n).index_add_(1, dst, cur[:, src] * w[None]) / den[None]
            own = self.node.weight[:n].expand(b, n, -1)
            return (nb_now.unsqueeze(-1) + self.head(torch.cat([own, agg, nb_now.unsqueeze(-1)], -1))).squeeze(-1)
        last = hist[..., -1:]
        return (last + self.head(torch.cat([z, agg, last], -1))).squeeze(-1)


def _windows(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = t[:, None] + np.arange(-HIST, 0)[None, :]
    return x[idx].transpose(0, 2, 1), x[t]  # (B, N, HIST), (B, N)


def train_stgnn(x: np.ndarray, t_train: np.ndarray, t_val: np.ndarray, edge_index: np.ndarray, w: np.ndarray, *,
                epochs: int = 6, batch: int = 32, lr: float = 2e-3, seed: int = 0, corrupt: float = 0.05,
                stride: int = 2, own_history: bool = True) -> tuple[STGNN, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = STGNN(own_history=own_history, n_nodes=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ei, ww = torch.as_tensor(edge_index), torch.as_tensor(w, dtype=torch.float32)
    t_train = t_train[t_train >= HIST][::stride]
    hist_curve = []
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(t_train), max(1, len(t_train) // batch)):
            hist, cur = _windows(x, b)
            vis = cur.copy()
            m = rng.random(vis.shape) < corrupt
            vis[m] = rng.uniform(5, 80, m.sum())
            opt.zero_grad()
            pred = model(torch.tensor(hist / SCALE), torch.tensor(vis / SCALE), ei, ww)
            loss = nn.functional.huber_loss(pred, torch.tensor(cur / SCALE), delta=0.1)
            loss.backward()
            opt.step()
        hist_curve.append(float(np.mean(np.abs(predict_stgnn(model, x, t_val[t_val >= HIST][::6], edge_index, w)
                                                 - x[t_val[t_val >= HIST][::6]]))))
    return model, hist_curve


@torch.no_grad()
def predict_stgnn(model: STGNN, x: np.ndarray, t: np.ndarray, edge_index: np.ndarray, w: np.ndarray,
                  batch: int = 256) -> np.ndarray:
    model.eval()
    ei, ww = torch.as_tensor(edge_index), torch.as_tensor(w, dtype=torch.float32)
    out = []
    for b in np.array_split(t, max(1, int(np.ceil(len(t) / batch)))):
        hist, cur = _windows(x, b)
        out.append(model(torch.tensor(hist / SCALE), torch.tensor(cur / SCALE), ei, ww).numpy() * SCALE)
    return np.concatenate(out)
