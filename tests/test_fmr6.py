import numpy as np

import fmr6


def test_model_loads_with_six_arm_joints_and_a_gripper():
    r = fmr6.Robot()
    assert r.model.nu == 7
    assert [r.model.joint(j).name for j in fmr6.JOINTS] == fmr6.JOINTS


def test_zero_pose_matches_the_drawing_dimensions():
    r = fmr6.Robot()
    pos, R = r.tcp(np.zeros(6))
    # J2 at 0.33 m, upper arm 0.58 m straight up, forearm 0.56 m + tool 0.262 m along +x
    assert np.allclose(pos, [0.56 + 0.262, 0.0, 0.33 + 0.58], atol=1e-6)
    assert np.allclose(R[:, 0], [1, 0, 0], atol=1e-6)


def test_wrist_axes_meet_at_one_point():
    r = fmr6.Robot()
    r.set_q(np.deg2rad([10, 20, -30, 40, 50, 60]))
    centres = [r.data.xpos[r.model.body(b).id] for b in ("link4", "link5", "link6")]
    assert np.allclose(centres[0], centres[1]) and np.allclose(centres[1], centres[2])


def test_ik_reaches_a_point_on_the_table_with_the_tool_pointing_down():
    r = fmr6.Robot("table")
    target = np.array([0.70, 0.05, 0.42])
    q = r.ik(target, tool_dir=(0, 0, -1))
    pos, R = r.tcp(q)
    assert np.linalg.norm(pos - target) < 1e-3
    assert R[:, 0] @ np.array([0, 0, -1]) > 0.999
