import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "labs" / "s0e3-action-tokens"))
import fmr6  # noqa: E402
import action_tokens as at  # noqa: E402


def test_the_videos_arithmetic():
    assert abs(200 / 256 - 0.78) < 0.005                # "20 cm over 256 bins is about 0.78 mm"


def test_round_trip_stays_within_one_bin():
    d = np.random.default_rng(0).normal(0, 0.003, (500, 3))
    st = at.statistics(d)
    inside = np.all((d >= st["q01"]) & (d <= st["q99"]), axis=1)
    err = np.abs(at.detokenize(at.tokenize(d, st), st) - d)[inside]
    assert np.all(err <= (st["q99"] - st["q01"]) / 255 / 2 + 1e-12)


def test_fewer_bins_cost_more_and_wrong_stats_barely_move():
    r = fmr6.Robot("table", arm="fr3")
    path = at.record_trajectory(r); d = np.diff(path, axis=0); st = at.statistics(d)
    ref = at.replay(r, path[0], d)
    err = {n: np.linalg.norm(at.replay(r, path[0], at.detokenize(at.tokenize(d, st, n), st, n))[-1] - ref[-1]) for n in (256, 4)}
    assert err[256] < 1e-3 and err[4] > err[256]
    wrong = at.replay(r, path[0], at.detokenize(at.tokenize(d, st), {"q01": st["q01"] / 20, "q99": st["q99"] / 20}))
    assert np.linalg.norm(wrong[-1] - wrong[0]) < 0.1 * np.linalg.norm(path[-1] - path[0])
