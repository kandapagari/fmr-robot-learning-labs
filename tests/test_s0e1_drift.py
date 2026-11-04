"""The numbers shown in the S0E1 video must come out of the lab code."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("drift", Path(__file__).parents[1] / "labs/s0e1-drift/drift.py")
drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift)


def test_open_loop_matches_the_video():
    p = drift.open_loop()
    assert abs(drift.offset_at(p, 1) - 0.0042) < 1e-4      # invisible at step 1
    assert abs(drift.offset_at(p, 50) - 5.34) < 0.01
    assert abs(drift.offset_at(p, 100) - 21.08) < 0.01     # "OFF BY 21.1 mm" at step 100


def test_feedback_keeps_the_error_small():
    assert drift.offset_at(drift.closed_loop(), 100) < 0.2 * drift.offset_at(drift.open_loop(), 100)


def test_world_path_is_reachable_by_fmr6():
    import fmr6
    r = fmr6.Robot("table")
    pts = drift.to_world(drift.open_loop()[::10])
    q = None
    for p in pts:
        q = r.ik(p, q0=q)
        assert np.linalg.norm(r.tcp(q)[0] - p) < 2e-3
