"""Thin, readable helpers around the FMR-6 MuJoCo model: forward kinematics, a damped least-squares IK,
and rendering. Kept small on purpose: students should be able to read all of it."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
# A comfortable starting pose (radians): upper arm leaning forward, forearm level, tool pointing down.
READY = np.deg2rad([0.0, 20.0, -10.0, 0.0, 80.0, 0.0])


def model_path(scene: str = "arm") -> Path:
    """'arm' → the robot alone, 'table' → robot in front of a table with a cup and a sponge."""
    return ASSETS / ("fmr6.xml" if scene == "arm" else f"scene_{scene}.xml")


class Robot:
    def __init__(self, scene: str = "arm"):
        self.model = mujoco.MjModel.from_xml_path(str(model_path(scene)))
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.dadr = np.array([self.model.joint(j).dofadr[0] for j in JOINTS])
        self.site = self.model.site("tcp").id
        self.lo = self.model.jnt_range[[self.model.joint(j).id for j in JOINTS], 0]
        self.hi = self.model.jnt_range[[self.model.joint(j).id for j in JOINTS], 1]
        self.set_q(READY)

    # ——— state ———
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def set_q(self, q) -> None:
        """Place the arm at joint angles q (radians) and update kinematics (no dynamics)."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def tcp(self, q=None):
        """Tool centre point: (position in metres, 3×3 rotation matrix). The tool axis is the matrix's first column."""
        if q is not None:
            self.set_q(q)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    # ——— inverse kinematics ———
    def ik(self, target_pos, tool_dir=(0.0, 0.0, -1.0), q0=None, iters: int = 200, damping: float = 1e-3,
           w_dir: float = 0.3, tol: float = 1e-5):
        """Damped least squares: move the tool tip to target_pos with the tool axis along tool_dir.
        Returns joint angles (radians). Starts from q0 (default: current pose)."""
        q = self.q() if q0 is None else np.array(q0, dtype=float)
        tool_dir = np.asarray(tool_dir, float) / np.linalg.norm(tool_dir)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            pos, R = self.tcp(q)
            e_pos = np.asarray(target_pos) - pos
            e_dir = np.cross(R[:, 0], tool_dir)                  # rotation that turns the tool axis onto tool_dir
            err = np.concatenate([e_pos, w_dir * e_dir])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], w_dir * jr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(q + dq, self.lo, self.hi)
        self.set_q(q)
        return q

    # ——— pictures ———
    def render(self, width: int = 640, height: int = 480, camera: dict | None = None) -> np.ndarray:
        """RGB image of the current pose. Headless machines: set MUJOCO_GL=egl (or osmesa) before importing mujoco."""
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (camera or {}).get("lookat", [0.45, 0.0, 0.55])
        cam.distance = (camera or {}).get("distance", 2.2)
        cam.azimuth = (camera or {}).get("azimuth", 135)
        cam.elevation = (camera or {}).get("elevation", -20)
        with mujoco.Renderer(self.model, height, width) as r:
            r.update_scene(self.data, cam)
            return r.render()
