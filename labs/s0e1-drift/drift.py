"""S0E1 lab: why small errors don't stay small when a robot acts in a loop.

A "policy" moves the FMR-6 tool tip along a straight 200 mm line in 100 steps of 2 mm.
Each step it picks a heading for the next 2 mm. The heading it outputs is off by a tiny bias
(0.12° per step), and because every step starts from where the last one ended, the bias adds up.

open_loop  : the policy never looks at where the tip actually is (like replaying a demonstration).
closed_loop: the policy measures its sideways offset and steers back toward the line.
"""
from __future__ import annotations

import numpy as np


def open_loop(steps: int = 100, step_mm: float = 2.0, bias_deg: float = 0.12) -> np.ndarray:
    """Tip positions (mm, shape (steps+1, 2): along-line x, sideways y) when the heading error accumulates."""
    p, heading, out = np.zeros(2), 0.0, [np.zeros(2)]
    for _ in range(steps):
        heading += np.deg2rad(bias_deg)                      # the same small mistake, every step
        p = p + step_mm * np.array([np.cos(heading), np.sin(heading)])
        out.append(p.copy())
    return np.array(out)


def closed_loop(steps: int = 100, step_mm: float = 2.0, bias_deg: float = 0.12, gain: float = 0.05) -> np.ndarray:
    """Same bias, but each step the policy also sees its sideways offset y and turns back toward the line."""
    p, heading, out = np.zeros(2), 0.0, [np.zeros(2)]
    for _ in range(steps):
        heading += np.deg2rad(bias_deg) - gain * p[1] / step_mm   # feedback: steer against the offset
        p = p + step_mm * np.array([np.cos(heading), np.sin(heading)])
        out.append(p.copy())
    return np.array(out)


def offset_at(path: np.ndarray, step: int) -> float:
    """Sideways distance from the planned straight line (mm) after a given step."""
    return float(abs(path[step, 1]))


def to_world(path_mm: np.ndarray, start=(0.62, -0.10, 0.40), direction=(0.0, 1.0)) -> np.ndarray:
    """Place a 2D path (mm) on the table in front of FMR-6: x along `direction`, y sideways. Returns (N, 3) metres."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    n = np.array([d[1], -d[0]])                                  # sideways direction on the table
    xy = np.asarray(start[:2]) + (path_mm[:, :1] * d + path_mm[:, 1:2] * n) / 1000.0
    return np.column_stack([xy, np.full(len(xy), start[2])])


if __name__ == "__main__":
    ol, cl = open_loop(), closed_loop()
    for k in (1, 10, 50, 100):
        print(f"step {k:3d}: open loop off by {offset_at(ol, k):7.3f} mm · closed loop {offset_at(cl, k):6.3f} mm")
