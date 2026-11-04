"""Thin, readable helpers around two MuJoCo arms: FMR-6 (the arm drawn in the videos) and the Franka FR3
(a real research arm, model from MuJoCo Menagerie). Forward kinematics, a damped least-squares IK, rendering.
Kept small on purpose: students should be able to read all of it."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
# A comfortable starting pose (radians): upper arm leaning forward, forearm level, tool pointing down.
READY = np.deg2rad([0.0, 20.0, -10.0, 0.0, 80.0, 0.0])


# Franka FR3 with the Franka Hand: 7 joints. Its home pose (Franka's own): elbow bent, hand pointing down.
FR3_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FR3_READY = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

ARMS = {
    #        files (arm alone, scene folder)   joints       ready pose  tool axis = column of the TCP rotation
    "fmr6": dict(arm="fmr6.xml", folder="", joints=JOINTS, ready=READY, tool_axis=0),
    "fr3": dict(arm="fr3/fr3_hand.xml", folder="fr3/", joints=FR3_JOINTS, ready=FR3_READY, tool_axis=2),
}


def model_path(scene: str = "arm", arm: str = "fmr6") -> Path:
    """'arm' → the robot alone, 'table' → robot in front of a table with a cup and a sponge."""
    a = ARMS[arm]
    return ASSETS / (a["arm"] if scene == "arm" else f"{a['folder']}scene_{scene}.xml")


class Robot:
    def __init__(self, scene: str = "arm", arm: str = "fmr6", model: mujoco.MjModel | None = None):
        """arm: "fmr6" (6 joints, the arm from the videos) or "fr3" (Franka FR3, 7 joints).
        model: a model you compiled yourself (e.g. the scene edited with mujoco.MjSpec); scene is then ignored."""
        cfg = ARMS[arm]
        self.arm, self.joints, self.ready, self.tool_axis = arm, cfg["joints"], cfg["ready"], cfg["tool_axis"]
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(model_path(scene, arm)))
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(j).id for j in self.joints]
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in self.joints])
        self.dadr = np.array([self.model.joint(j).dofadr[0] for j in self.joints])
        self.site = self.model.site("tcp").id
        self.lo, self.hi = self.model.jnt_range[ids, 0], self.model.jnt_range[ids, 1]
        if self.model.nkey:                                   # FR3: start with the fingers open
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.set_q(self.ready)

    # ——— state ———
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def set_q(self, q) -> None:
        """Place the arm at joint angles q (radians) and update kinematics (no dynamics)."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)            # lights and cameras follow the bodies too

    def tcp(self, q=None):
        """Tool centre point: (position in metres, 3×3 rotation matrix). The tool axis is column `self.tool_axis`
        (0 for FMR-6, 2 for the FR3)."""
        if q is not None:
            self.set_q(q)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    # ——— inverse kinematics ———
    def ik(self, target_pos, tool_dir=(0.0, 0.0, -1.0), q0=None, iters: int = 200, damping: float = 1e-3,
           w_dir: float = 0.3, tol: float = 1e-5, grip_dir=None):
        """Damped least squares: move the tool tip to target_pos with the tool axis along tool_dir.
        Returns joint angles (radians). Starts from q0 (default: current pose).
        With 7 joints (FR3) there are many solutions; starting near the ready pose keeps the elbow sensible.
        grip_dir: optional direction the fingers should close along (the TCP's y axis), e.g. across a narrow object.
        Without it the hand may end up at any rotation about the tool axis."""
        q = self.q() if q0 is None else np.array(q0, dtype=float)
        tool_dir = np.asarray(tool_dir, float) / np.linalg.norm(tool_dir)
        if grip_dir is not None:
            grip_dir = np.asarray(grip_dir, float) / np.linalg.norm(grip_dir)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            pos, R = self.tcp(q)
            e_pos = np.asarray(target_pos) - pos
            e_dir = np.cross(R[:, self.tool_axis], tool_dir)                  # rotation that turns the tool axis onto tool_dir
            if grip_dir is not None:                                          # and the finger axis onto grip_dir
                e_dir = e_dir + np.cross(R[:, 1], grip_dir)
            err = np.concatenate([e_pos, w_dir * e_dir])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], w_dir * jr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)        # 6×n Jacobian, n = 6 or 7
            q = np.clip(q + dq, self.lo, self.hi)
        self.set_q(q)
        return q

    # ——— pictures ———
    def render(self, width: int = 640, height: int = 480, camera: dict | None = None, trail=None,
               trail_rgba=(0.85, 0.1, 0.1, 1.0), guide=None) -> np.ndarray:
        """RGB image of the current pose. trail: optional (N, 3) points (m) drawn as a thin red line, e.g. the tool
        tip's path. guide: optional (N, 3) points drawn as a grey line, e.g. the planned path.
        Headless machines: set MUJOCO_GL=egl (or osmesa) before importing mujoco."""
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (camera or {}).get("lookat", [0.45, 0.0, 0.55])
        cam.distance = (camera or {}).get("distance", 2.2)
        cam.azimuth = (camera or {}).get("azimuth", 135)
        cam.elevation = (camera or {}).get("elevation", -20)
        with mujoco.Renderer(self.model, height, width) as r:
            r.update_scene(self.data, cam)
            lines = [(guide, (0.35, 0.38, 0.42, 1.0)), (trail, trail_rgba)]
            for pts, rgba in [(np.asarray(p, float), c) for p, c in lines if p is not None]:
                for a, b in zip(pts[:-1], pts[1:]):
                    if r.scene.ngeom >= r.scene.maxgeom:
                        break
                    g = r.scene.geoms[r.scene.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                        np.asarray(rgba, np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025, a, b)
                    r.scene.ngeom += 1
            return r.render()
