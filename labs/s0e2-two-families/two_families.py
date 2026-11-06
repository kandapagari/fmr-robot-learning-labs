"""S0E2 lab: one reaching task, two families of action model.

classical : inverse kinematics. Derives joint angles from a model of the arm; no data.
imitation : a policy that only ever sees demonstrations (target → joint angles) and copies them.
            It interpolates between nearby demos with a small local linear fit, and it has no access to IK.

The demos all come from one half of the table. Inside that half the copier is nearly as good as IK.
Outside it, it has to guess, and the guess gets worse the further you go.
"""
from __future__ import annotations

import numpy as np

TABLE_X = (0.45, 0.70)          # metres in front of the robot base
LEFT_Y, RIGHT_Y = (-0.25, 0.0), (0.0, 0.25)
HEIGHT = 0.45                   # fingertips 7 cm above the table top (0.38 m), hand pointing down


def sample_targets(n: int, y_range=LEFT_Y, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(*TABLE_X, n), rng.uniform(*y_range, n), np.full(n, HEIGHT)])


def demonstrate(robot, targets) -> np.ndarray:
    """What a demonstrator would record: the joint angles that put the hand on each target (here we cheat and use IK)."""
    return np.array([robot.ik(t, q0=robot.ready) for t in targets])


class CopyDemos:
    """Imitation in miniature: predict joint angles for a new target from the k nearest demonstrations,
    with a local linear fit (weighted least squares). Nothing here knows how the arm is built."""

    def __init__(self, k: int = 8):
        self.k = k

    def fit(self, targets, joint_angles):
        self.X, self.Q = np.asarray(targets, float), np.asarray(joint_angles, float)
        return self

    def __call__(self, target) -> np.ndarray:
        d = np.linalg.norm(self.X[:, :2] - np.asarray(target)[:2], axis=1)
        idx = np.argsort(d)[: self.k]
        w = 1.0 / (d[idx] + 1e-3)
        A = np.column_stack([np.ones(len(idx)), self.X[idx, :2]])            # [1, x, y]; height is fixed
        W = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A * W, self.Q[idx] * W, rcond=None)
        return np.array([1.0, *np.asarray(target)[:2]]) @ coef

    def distance_to_demos(self, target) -> float:
        return float(np.min(np.linalg.norm(self.X[:, :2] - np.asarray(target)[:2], axis=1)))


def reach_error(robot, q, target) -> float:
    """How far (m) the fingertips end up from the target with joint angles q."""
    return float(np.linalg.norm(robot.tcp(q)[0] - np.asarray(target)))


def compare(robot, policy, targets):
    """Error of IK and of the copier on each target, plus how far each target is from the nearest demo."""
    rows = []
    for t in targets:
        rows.append((reach_error(robot, robot.ik(t, q0=robot.ready), t), reach_error(robot, policy(t), t), policy.distance_to_demos(t)))
    return np.array(rows)     # columns: ik_error, copy_error, distance_to_nearest_demo (all metres)
