"""Sensor tampering injected into real PEMS-BAY speeds.

==========  ======================================================================
attack      what the compromised detector reports during the event (1-3 h)
==========  ======================================================================
fake_jam    a queue that is not there: 8-20 mph with noise (e.g. to divert traffic or trigger ramp metering)
stuck       its value at event onset, frozen (a hung or spoofed controller)
drift       a bias growing linearly to -25 mph over the event (slow, hard to see)
noise       the true speed plus heavy noise (sd 15 mph)
replay      its own readings from exactly 24 h earlier (plausible, but not today's traffic)
==========  ======================================================================

Real congestion is left untouched, so the test period contains both genuine jams and fake ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ATTACKS = ("fake_jam", "stuck", "drift", "noise", "replay")
STEPS_PER_DAY = 288


@dataclass(frozen=True)
class Event:
    sensor: int
    start: int
    end: int
    kind: str


def inject(speed: np.ndarray, t_range: np.ndarray, *, n_events: int = 150, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[Event]]:
    """Returns (tampered copy, label mask (T, N) of tampered samples, events). Events never overlap on a sensor."""
    rng = np.random.default_rng(seed)
    out = speed.copy()
    label = np.zeros(speed.shape, bool)
    events: list[Event] = []
    lo, hi = int(t_range.min()) + STEPS_PER_DAY, int(t_range.max()) - 40
    tries = 0
    while len(events) < n_events and tries < 50 * n_events:
        tries += 1
        s = int(rng.integers(0, speed.shape[1]))
        start = int(rng.integers(lo, hi))
        dur = int(rng.integers(12, 37))  # 1-3 h
        end = start + dur
        if label[max(start - 12, 0):end + 12, s].any():
            continue
        kind = ATTACKS[len(events) % len(ATTACKS)]
        sl = slice(start, end)
        true = speed[sl, s]
        if kind == "fake_jam":
            out[sl, s] = rng.uniform(8, 20) + rng.normal(0, 2, dur)
        elif kind == "stuck":
            out[sl, s] = speed[start - 1, s]
        elif kind == "drift":
            out[sl, s] = true - np.linspace(0, 25, dur)
        elif kind == "noise":
            out[sl, s] = true + rng.normal(0, 15, dur)
        elif kind == "replay":
            out[sl, s] = speed[start - STEPS_PER_DAY:end - STEPS_PER_DAY, s]
        out[sl, s] = np.clip(out[sl, s], 3, 85)
        label[sl, s] = True
        events.append(Event(s, start, end, kind))
    return out, label, events
