"""Scoring API for a traffic management centre's data pipeline.

``POST /v1/check``  the last 13 five-minute speed snapshots for all sensors (12 history + current), in the
                    model's sensor order. Returns, per sensor, the speed the network predicts from its own
                    history and its neighbours, the residual, and whether it exceeds that sensor's
                    calibrated threshold. One snapshot is a single-step check; the study smooths residuals
                    over 15 minutes before alarming, which callers should do too.
``GET  /health``
"""

from __future__ import annotations

import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__
from ..models import HIST, STGNN, predict_stgnn

DEFAULT_THRESHOLD_MPH = 10.0


class Window(BaseModel):
    speeds: list[list[float]] = Field(description="(13, N) mph, oldest first; the last row is the current snapshot")


def create_app(model_path: str | None = None, bundle: dict | None = None) -> FastAPI:
    if bundle is None:
        bundle = torch.load(model_path or os.environ.get("SENSORGUARD_MODEL", "models/stgnn.pt"), weights_only=False)
    ei, w = np.asarray(bundle["edge_index"]), np.asarray(bundle["edge_weight"])
    sensors = np.asarray(bundle["sensors"])
    model = STGNN(own_history=bool(bundle.get("own_history", True)), n_nodes=len(sensors))
    model.load_state_dict(bundle["state"])
    thr = np.asarray(bundle.get("threshold_mph", np.full(len(sensors), DEFAULT_THRESHOLD_MPH)))
    app = FastAPI(title="Sensor tamper check", version=__version__)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "sensors": len(sensors)}

    @app.post("/v1/check")
    def check(win: Window) -> dict[str, object]:
        x = np.asarray(win.speeds, dtype=np.float32)
        if x.shape != (HIST + 1, len(sensors)):
            raise HTTPException(422, f"expected shape ({HIST + 1}, {len(sensors)}), got {list(x.shape)}")
        if not np.isfinite(x).all() or (x < 0).any():
            raise HTTPException(422, "speeds must be finite and non-negative")
        pred = predict_stgnn(model, x, np.array([HIST]), ei, w)[0]
        r = np.abs(x[HIST] - pred)
        flagged = np.flatnonzero(r > thr)
        return {"flagged": [{"sensor": int(sensors[i]), "reported": float(x[HIST, i]), "predicted": round(float(pred[i]), 1),
                             "residual": round(float(r[i]), 1)} for i in flagged[np.argsort(-r[flagged])]],
                "max_residual": round(float(r.max()), 1)}

    return app
