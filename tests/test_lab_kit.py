"""The physics claims the Lab Kit videos make: the FR3 can pick the sponge across its width but not along its length."""
import mujoco
import numpy as np

import fmr6


def pick(grip_dir):
    r = fmr6.Robot("table", arm="fr3")
    m, d = r.model, r.data
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    sponge = d.body("sponge").xpos.copy()
    above = r.ik(sponge + [0, 0, 0.08], q0=r.ready, grip_dir=grip_dir)
    grasp = r.ik(sponge + [0, 0, 0.005], q0=r.ready, grip_dir=grip_dir)
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    for q_target, gripper, seconds in [(above, 255, 1.5), (grasp, 255, 1.0), (grasp, 0, 0.8), (above, 0, 1.5)]:
        start, n = d.ctrl[:7].copy(), int(seconds / m.opt.timestep)
        for i in range(n):
            s = min(1.0, (i + 1) / (0.7 * n)); s = s * s * (3 - 2 * s)
            d.ctrl[:7] = start + s * (q_target - start); d.ctrl[7] = gripper
            mujoco.mj_step(m, d)
    return d.body("sponge").xpos[2] - sponge[2]


def test_fr3_lifts_the_sponge_gripped_across_its_width():
    assert pick((0, 1, 0)) > 0.05


def test_fr3_cannot_grip_the_sponge_along_its_length():
    assert pick((1, 0, 0)) < 0.01                             # 10 cm sponge, 8 cm maximum opening


def test_user_compiled_model_is_accepted():
    spec = mujoco.MjSpec.from_file(str(fmr6.model_path("table", arm="fr3")))
    b = spec.worldbody.add_body(name="block", pos=[0.6, 0.15, 0.55]); b.add_freejoint()
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025] * 3, mass=0.1)
    r = fmr6.Robot(arm="fr3", model=spec.compile())
    assert r.model.body("block").id > 0 and len(r.joints) == 7
