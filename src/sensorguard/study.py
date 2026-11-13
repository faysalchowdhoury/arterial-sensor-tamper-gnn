"""Experiment: real PEMS-BAY traffic, injected tampering, three detectors."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .data import load, split  # noqa: E402
from .models import HIST, STGNN, predict_persistence, predict_spatial, predict_stgnn, train_stgnn  # noqa: E402
from .tamper import ATTACKS, inject  # noqa: E402

METHODS = ("persistence", "spatial", "stgnn", "stgnn_nb")
SMOOTH = 3  # 15-min causal mean of residuals
Q = 0.999
FLOOR_MPH = 5.0


def causal_mean(r: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    c = np.cumsum(np.vstack([np.zeros((1, r.shape[1])), r]), axis=0)
    out = np.empty_like(r)
    for i in range(r.shape[0]):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def predict(method: str, x: np.ndarray, t: np.ndarray, ds_ei: np.ndarray, ds_w: np.ndarray, models: dict[str, STGNN]) -> np.ndarray:
    if method == "persistence":
        return predict_persistence(x, t)
    if method == "spatial":
        return predict_spatial(x, t, ds_ei, ds_w)
    return predict_stgnn(models[method], x, t, ds_ei, ds_w)


def run(data_dir: str, out_dir: str | Path, *, seed: int = 0, epochs: int = 6, n_events: int = 150,
        save_model: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ds = load(data_dir)
    sp = split(ds.times)
    x = ds.speed
    t0 = time.perf_counter()
    model, curve = train_stgnn(x, sp["train"], sp["val"], ds.edge_index, ds.edge_weight, epochs=epochs, seed=seed)
    fit_s = time.perf_counter() - t0
    model_nb, curve_nb = train_stgnn(x, sp["train"], sp["val"], ds.edge_index, ds.edge_weight, epochs=epochs, seed=seed,
                                     own_history=False)
    models = {"stgnn": model, "stgnn_nb": model_nb}

    tv = sp["val"][sp["val"] >= HIST]
    tt = sp["test"]
    xt, label, events = inject(x, tt, n_events=n_events, seed=seed)
    free_flow = np.quantile(x[sp["train"]], 0.85, axis=0)
    congested = x[tt] < 0.6 * free_flow  # real congestion (true speed), whether or not tampered
    adj = np.zeros((x.shape[1], x.shape[1]), bool)
    adj[ds.edge_index[0], ds.edge_index[1]] = True
    near = np.zeros((len(tt), x.shape[1]), bool)  # clean sensors next to an active tampering event
    for e in events:
        near[e.start - tt[0]:e.end - tt[0], adj[e.sensor]] = True
    lab = label[tt]
    clean = ~lab
    days = len(tt) / 288

    res: dict[str, Any] = {"methods": {}, "events": len(events), "test_days": days, "fit_s": fit_s, "val_mae_curve": curve,
                           "val_mae_curve_nb": curve_nb}
    scores: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    for m in METHODS:
        t1 = time.perf_counter()
        pv = predict(m, x, tv, ds.edge_index, ds.edge_weight, models)
        thr = np.maximum(np.quantile(causal_mean(np.abs(x[tv] - pv)), Q, axis=0), FLOOR_MPH)
        pt = predict(m, xt, tt, ds.edge_index, ds.edge_weight, models)
        infer_s = time.perf_counter() - t1
        s = causal_mean(np.abs(xt[tt] - pt))
        alarm = s > thr
        scores[m], preds[m] = s, pt
        if m == "stgnn_nb" and save_model:
            Path(save_model).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model_nb.state_dict(), "own_history": False, "edge_index": ds.edge_index, "edge_weight": ds.edge_weight,
                        "sensors": ds.sensors, "threshold_mph": thr}, save_model)
        per: dict[str, Any] = {}
        for a in ATTACKS:
            evs = [e for e in events if e.kind == a]
            ttd = []
            for e in evs:
                hit = np.flatnonzero(alarm[e.start - tt[0]:e.end - tt[0], e.sensor])
                ttd.append(float(hit[0] * 5) if len(hit) else None)
            found = [d for d in ttd if d is not None]
            msk = np.zeros_like(lab)
            for e in evs:
                msk[e.start - tt[0]:e.end - tt[0], e.sensor] = True
            y = np.r_[np.zeros(clean.sum()), np.ones(msk.sum())]
            per[a] = {"n": len(evs), "detected": len(found) / max(len(evs), 1),
                      "median_ttd_min": float(np.median(found)) if found else None,
                      "auc": float(roc_auc_score(y, np.r_[s[clean], s[msk]]))}
        episodes = int(((alarm & clean)[1:] & ~(alarm & clean)[:-1]).sum())
        res["methods"][m] = {
            "attacks": per,
            "val_mae_mph": float(np.mean(np.abs(x[tv] - pv))),
            "false_alarm_rate": float(alarm[clean].mean()),
            "false_alarm_rate_real_congestion": float(alarm[clean & congested].mean()),
            "false_alarm_rate_free_flow": float(alarm[clean & ~congested].mean()),
            "false_alarm_rate_next_to_tampered": float(alarm[clean & near].mean()),
            "false_alarm_episodes_per_sensor_day": episodes / (x.shape[1] * days),
            "inference_ms_per_snapshot": 1000 * infer_s / (len(tv) + len(tt)),
        }
    res["congested_share_of_test"] = float(congested.mean())
    res["graph"] = {"nodes": int(x.shape[1]), "edges": int(ds.edge_index.shape[1] // 2)}
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _plot(res, xt, x, tt, events, preds, scores, congested, out / "figures")
    write_report(res, out / "RESULTS.md")
    return res


def _plot(res: dict[str, Any], xt: np.ndarray, x: np.ndarray, tt: np.ndarray, events: list, preds: dict[str, np.ndarray],
          scores: dict[str, np.ndarray], congested: np.ndarray, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1.0, 1.2, 1.2]})
    mat = np.array([[res["methods"][m]["attacks"][a]["detected"] for m in METHODS] for a in ATTACKS])
    ax = axes[0]
    ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METHODS)), METHODS)
    ax.set_yticks(range(len(ATTACKS)), ATTACKS)
    for i in range(len(ATTACKS)):
        for j in range(len(METHODS)):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", color="w" if mat[i, j] > 0.6 else "k", fontsize=9)
    ax.set_title("Tampering events detected")
    hours = np.arange(-24, 60) / 12
    fj = next(e for e in events if e.kind == "fake_jam")
    ax = axes[1]
    k = np.arange(fj.start - 24, fj.start + 60) - tt[0]
    ax.plot(hours, x[tt[k], fj.sensor], "k", lw=1, label="true speed")
    ax.plot(hours, xt[tt[k], fj.sensor], "r", lw=1.2, label="reported (fake jam)")
    for m, c in (("persistence", "#9e9e9e"), ("stgnn", "#1565c0"), ("stgnn_nb", "#2e7d32")):
        ax.plot(hours, preds[m][k, fj.sensor], color=c, lw=1, ls="--", label=f"{m} prediction")
    ax.axvspan(0, (fj.end - fj.start) / 12, color="r", alpha=0.08)
    ax.set_xlabel("hours from tampering onset")
    ax.set_ylabel("mph")
    ax.set_title(f"Fake jam on sensor {fj.sensor}")
    ax.legend(fontsize=7)
    # a real congestion onset on an untampered sensor
    tampered = {e.sensor for e in events}
    cand = [(i, j) for i, j in zip(*np.nonzero(congested[24:-60] & ~congested[23:-61])) if j not in tampered]
    i0, s0 = cand[len(cand) // 2]
    i0 += 24
    ax = axes[2]
    k2 = np.arange(i0 - 24, i0 + 60)
    ax.plot(hours, x[tt[k2], s0], "k", lw=1.2, label="true = reported speed")
    for m, c in (("persistence", "#9e9e9e"), ("stgnn", "#1565c0"), ("stgnn_nb", "#2e7d32")):
        ax.plot(hours, preds[m][k2, s0], color=c, lw=1, ls="--", label=f"{m} prediction")
    ax.set_xlabel("hours from congestion onset")
    ax.set_title(f"Real congestion on sensor {s0} (not tampered)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "detection.png", dpi=150)
    plt.close(fig)


def write_report(r: dict[str, Any], path: Path) -> Path:
    lines = ["# Sensor tampering vs real congestion on PEMS-BAY", "",
             f"{r['graph']['nodes']} sensors, {r['graph']['edges']} graph edges; {r['events']} tampering events (1-3 h) injected "
             f"into {r['test_days']:.0f} test days (May-Jun 2017); {r['congested_share_of_test']:.1%} of test samples are real congestion.", "",
             "## Events detected / median time to detect (min) / sample AUC", "",
             "| Attack | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for a in ATTACKS:
        cells = []
        for m in METHODS:
            p = r["methods"][m]["attacks"][a]
            ttd = "-" if p["median_ttd_min"] is None else f"{p['median_ttd_min']:.0f}"
            cells.append(f"{p['detected']:.0%} / {ttd} / {p['auc']:.3f}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", "## False alarms on clean samples (thresholds: per-sensor 99.9th percentile of clean April scores)", "",
              "| Method | Prediction MAE (mph, Apr) | All clean | In real congestion | In free flow | Next to a tampered sensor | Episodes per sensor-day |",
              "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        p = r["methods"][m]
        lines.append(f"| {m} | {p['val_mae_mph']:.2f} | {p['false_alarm_rate']:.3%} | {p['false_alarm_rate_real_congestion']:.3%} | "
                     f"{p['false_alarm_rate_free_flow']:.3%} | {p['false_alarm_rate_next_to_tampered']:.3%} | "
                     f"{p['false_alarm_episodes_per_sensor_day']:.3f} |")
    lines += ["", f"STGNN training: {r['fit_s'] / 60:.1f} min on CPU; validation MAE by epoch: "
              + ", ".join(f"{v:.2f}" for v in r["val_mae_curve"]) + "; neighbours-only: "
              + ", ".join(f"{v:.2f}" for v in r["val_mae_curve_nb"]), "", "![Detection](figures/detection.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
