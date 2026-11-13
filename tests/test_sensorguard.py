from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from sensorguard.api.app import create_app
from sensorguard.data import build_graph, split
from sensorguard.models import HIST, STGNN, neighbour_mean, predict_spatial, predict_stgnn, train_stgnn
from sensorguard.study import causal_mean
from sensorguard.tamper import ATTACKS, inject


def _meta() -> pd.DataFrame:
    return pd.DataFrame({"Fwy": [101] * 4 + [280] * 2, "Dir": ["N"] * 4 + ["S"] * 2,
                         "Abs_PM": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0],
                         "Latitude": [37.0, 37.01, 37.02, 37.03, 37.5, 37.51],
                         "Longitude": [-122.0, -122.0, -122.0, -122.0, -121.0, -121.0]})


def test_graph_links_along_freeway_and_within_radius() -> None:
    ei, w, g = build_graph(_meta(), k_along=2, cross_km=1.0)
    assert g.has_edge(0, 1) and g.has_edge(0, 2) and not g.has_edge(0, 3)  # two nearest along 101 N
    assert g.has_edge(4, 5) and not g.has_edge(3, 4)  # different freeway, far apart
    assert ei.shape[1] == 2 * g.number_of_edges() and (w > 0).all() and (w <= 1).all()


def test_neighbour_mean_and_spatial_prediction() -> None:
    ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    w = np.ones(4)
    x = np.array([[60.0, 60.0, 60.0], [30.0, 30.0, 30.0]])
    assert np.allclose(neighbour_mean(x, ei, w, 3)[1], [30.0, 30.0, 30.0])
    # sensor 1's neighbours halved their speed, so the spatial predictor expects sensor 1 to halve too
    assert np.allclose(predict_spatial(x, np.array([1]), ei, w)[0], [30.0, 30.0, 30.0])


def test_tamper_injection() -> None:
    rng = np.random.default_rng(0)
    x = (60 + rng.normal(0, 2, (3000, 20))).astype(np.float32)
    out, lab, events = inject(x, np.arange(1000, 3000), n_events=20, seed=1)
    assert len(events) == 20 and {e.kind for e in events} == set(ATTACKS)
    assert np.array_equal(out[~lab], x[~lab]) and (out[lab] != x[lab]).mean() > 0.8
    fj = next(e for e in events if e.kind == "fake_jam")
    assert out[fj.start:fj.end, fj.sensor].max() < 30
    st = next(e for e in events if e.kind == "stuck")
    assert np.ptp(out[st.start:st.end, st.sensor]) == 0


def test_stgnn_does_not_see_own_current_value() -> None:
    ei = torch.tensor([[0, 1], [1, 0]])
    w = torch.ones(2)
    nb = STGNN(own_history=False, n_nodes=2)
    h1, h2 = torch.rand(1, 2, HIST), torch.rand(1, 2, HIST)
    h2[0, 1] = h1[0, 1]  # change node 0's history only
    c = torch.tensor([[0.5, 0.5]])
    assert torch.isclose(nb(h1, c, ei, w)[0, 0], nb(h2, c, ei, w)[0, 0])  # neighbours-only ignores its own history
    m = STGNN(n_nodes=2)
    hist = torch.rand(1, 2, HIST)
    a = m(hist, torch.tensor([[0.5, 0.5]]), ei, w)
    b = m(hist, torch.tensor([[0.9, 0.5]]), ei, w)  # change node 0's current reading only
    assert torch.isclose(a[0, 0], b[0, 0]) and not torch.isclose(a[0, 1], b[0, 1])


def test_stgnn_learns_a_coherent_network() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(2000)
    base = 60 - 30 * (np.sin(t / 40) > 0.6)  # network-wide congestion waves
    x = (base[:, None] + rng.normal(0, 1, (2000, 6))).astype(np.float32)
    ei, w, _ = build_graph(_meta())
    model, curve = train_stgnn(x, np.arange(HIST, 1500), np.arange(1500, 2000), ei, w, epochs=3, stride=1)
    tt = np.arange(1500, 2000)
    mae = np.abs(predict_stgnn(model, x, tt, ei, w) - x[tt]).mean()
    assert mae < np.abs(x[tt - 1] - x[tt]).mean()  # better than persistence: congestion onsets come from neighbours


def test_split_and_causal_mean() -> None:
    times = pd.date_range("2017-01-01", "2017-06-30 23:55", freq="5min")
    s = split(times)
    assert times[s["val"]].month.unique().tolist() == [4] and times[s["test"]].month.min() == 5
    r = np.array([[3.0], [0.0], [0.0], [6.0]])
    assert np.allclose(causal_mean(r)[:, 0], [3.0, 1.5, 1.0, 2.0])


def test_api_flags_inconsistent_sensor() -> None:
    ei, w, _ = build_graph(_meta())
    model = STGNN(n_nodes=6)
    bundle = {"state": model.state_dict(), "edge_index": ei, "edge_weight": w, "sensors": np.arange(100, 106),
              "threshold_mph": np.full(6, 10.0)}
    cl = TestClient(create_app(bundle=bundle))
    win = np.full((HIST + 1, 6), 60.0)
    win[-1, 2] = 5.0  # one sensor suddenly reports a jam nobody else sees
    r = cl.post("/v1/check", json={"speeds": win.tolist()}).json()
    pred = predict_stgnn(model, win.astype(np.float32), np.array([HIST]), ei, w)[0]
    assert {f["sensor"] for f in r["flagged"]} == {int(100 + i) for i in np.flatnonzero(np.abs(win[-1] - pred) > 10)}
    assert 102 in [f["sensor"] for f in r["flagged"]]
    assert cl.post("/v1/check", json={"speeds": win[:5].tolist()}).status_code == 422
