"""Command line: ``sensorguard study`` and ``sensorguard serve``."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sensorguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("study", help="train on Jan-Mar, calibrate on Apr, test on tampered May-Jun")
    st.add_argument("--data", required=True, help="directory with PEMS-BAY.csv and PEMS-BAY-META.csv")
    st.add_argument("--out", default="results/pems_bay")
    st.add_argument("--epochs", type=int, default=6)
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--save-model", default="models/stgnn.pt")
    sv = sub.add_parser("serve", help="run the scoring API")
    sv.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    if a.cmd == "study":
        from .study import run

        r = run(a.data, a.out, seed=a.seed, epochs=a.epochs, save_model=a.save_model)
        print(json.dumps({m: {k: v for k, v in p.items() if k != "attacks"} for m, p in r["methods"].items()}, indent=2))
        return 0
    import uvicorn

    uvicorn.run("sensorguard.api.app:create_app", factory=True, host="0.0.0.0", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
