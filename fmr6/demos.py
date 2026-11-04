"""Small demos used in the Lab Kit videos. Read these after the lessons that explain them."""
from __future__ import annotations

import mujoco
import numpy as np


def spare_joint_sweep(robot, target, steps: int = 150, step: float = 0.02, camera: dict | None = None,
                      size=(480, 300)):
    """Move a 7-joint arm along its "spare" direction: the one way to turn the joints that doesn't move the hand.
    That direction is the last right-singular vector of the 6×7 Jacobian (A3 covers Jacobians).
    Returns (frames, worst hand drift in m, largest elbow travel in m)."""
    m, d = robot.model, robot.data
    q = robot.ik(target, q0=robot.ready, grip_dir=(0, 1, 0))
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    elbow_body = m.jnt_bodyid[m.joint(robot.joints[3]).id]                      # the link joint 4 turns
    frames, elbows, drift, prev = [], [], 0.0, None
    for _ in range(steps):
        robot.set_q(q)
        mujoco.mj_jacSite(m, d, jp, jr, robot.site)
        n = np.linalg.svd(np.vstack([jp[:, robot.dadr], jr[:, robot.dadr]]))[2][-1]
        n = n * np.sign(n[2]) if prev is None else (n if n @ prev >= 0 else -n)   # keep going the same way
        prev = n
        nxt = q + step * n
        if np.any(nxt < robot.lo) or np.any(nxt > robot.hi):                      # stop at a joint limit
            break
        q = robot.ik(target, q0=nxt, grip_dir=(0, 1, 0), iters=5)                 # tidy up tiny drift
        elbows.append(d.xpos[elbow_body].copy())
        drift = max(drift, float(np.linalg.norm(robot.tcp()[0] - np.asarray(target))))
        if camera is not None:
            frames.append(robot.render(*size, camera=camera))
    elbows = np.array(elbows)
    return frames, drift, float(np.linalg.norm(elbows - elbows[0], axis=1).max())
