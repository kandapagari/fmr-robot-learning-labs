"""FMR-6: the robot arm from the Foundation Models for Robotics videos, as a MuJoCo model plus small helpers.

    import fmr6
    robot = fmr6.Robot()                 # arm alone; fmr6.Robot("table") for the table scene
    robot.set_q(fmr6.READY)
    print(robot.tcp())                   # tool centre point position (m) and rotation matrix
    q = robot.ik([0.75, 0.0, 0.55])      # joint angles that put the tool tip there (tool pointing down)

    fr3 = fmr6.Robot("table", arm="fr3")  # the same table scene with a Franka FR3 + Franka Hand
"""
from .demos import spare_joint_sweep
from .robot import ARMS, ASSETS, FR3_JOINTS, FR3_READY, JOINTS, READY, Robot, model_path

__all__ = ["Robot", "READY", "JOINTS", "FR3_READY", "FR3_JOINTS", "ARMS", "ASSETS", "model_path", "spare_joint_sweep"]
__version__ = "0.2.0"
