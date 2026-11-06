"""S0E3 lab: OpenVLA-style action tokens, built by hand.

OpenVLA turns every continuous action number into one of 256 bins, spaced evenly between the 1st and 99th
percentile of that number in the training data, and gives each bin a token of its own. Here you do the same to
a recorded FR3 trajectory and measure what the bins cost.
"""
from __future__ import annotations

import numpy as np


def record_trajectory(robot, start=(0.70, -0.15, 0.45), end=(0.55, 0.15, 0.50), steps: int = 120) -> np.ndarray:
    """A smooth hand path (a gentle S-curve between two points), as fingertip positions in metres, shape (steps+1, 3)."""
    s = np.linspace(0, 1, steps + 1)[:, None]
    a, b = np.asarray(start, float), np.asarray(end, float)
    path = a + (b - a) * (3 * s ** 2 - 2 * s ** 3)
    path[:, 0] += 0.04 * np.sin(2 * np.pi * s[:, 0])            # a sideways wiggle so x isn't monotonic
    return path


def statistics(deltas: np.ndarray) -> dict:
    """Per-axis 1st and 99th percentiles: what OpenVLA stores as its normalisation statistics."""
    return {'q01': np.percentile(deltas, 1, axis=0), 'q99': np.percentile(deltas, 99, axis=0)}


def tokenize(deltas, stats, n_bins: int = 256) -> np.ndarray:
    lo, hi = stats['q01'], stats['q99']
    u = (np.clip(deltas, lo, hi) - lo) / np.where(hi > lo, hi - lo, 1)
    return np.round(u * (n_bins - 1)).astype(int)                 # 0 … n_bins-1


def detokenize(bins, stats, n_bins: int = 256) -> np.ndarray:
    lo, hi = stats['q01'], stats['q99']
    return lo + np.asarray(bins) / (n_bins - 1) * (hi - lo)


def bin_width_mm(stats, n_bins: int = 256) -> np.ndarray:
    return (stats['q99'] - stats['q01']) / (n_bins - 1) * 1000


def replay(robot, start, deltas) -> np.ndarray:
    """Run the deltas open-loop: each step asks IK for the hand at (last commanded point + delta); returns where the hand went."""
    target, q, out = np.asarray(start, float).copy(), None, []
    q = robot.ik(target, q0=robot.ready)
    out.append(robot.tcp(q)[0])
    for d in deltas:
        target = target + d
        q = robot.ik(target, q0=q)
        out.append(robot.tcp(q)[0])
    return np.array(out)
