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


# ——— Franka FR3 ———
def test_fr3_loads_with_seven_joints_a_hand_and_points_down_at_home():
    r = fmr6.Robot(arm="fr3")
    assert len(r.joints) == 7 and r.model.nu == 8           # 7 arm actuators + gripper
    pos, R = r.tcp()
    assert np.allclose(pos, [0.5545, 0.0, 0.5211], atol=2e-3)  # Franka's home: TCP 0.55 m out, 0.52 m up
    assert R[:, 2] @ np.array([0, 0, -1]) > 0.999


def test_fr3_ik_follows_the_s0e1_line_on_the_table():
    r = fmr6.Robot("table", arm="fr3")
    q = None
    for y in np.linspace(-0.10, 0.10, 11):                   # the 200 mm line from the S0E1 lab
        target = np.array([0.62, y, 0.40])
        q = r.ik(target, tool_dir=(0, 0, -1), q0=q)
        pos, R = r.tcp(q)
        assert np.linalg.norm(pos - target) < 1e-3
        assert R[:, 2] @ np.array([0, 0, -1]) > 0.999
        assert np.all(q >= r.lo - 1e-9) and np.all(q <= r.hi + 1e-9)
