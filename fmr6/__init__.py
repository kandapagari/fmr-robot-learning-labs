"""FMR-6: the robot arm from the Foundation Models for Robotics videos, as a MuJoCo model plus small helpers.

    import fmr6
    robot = fmr6.Robot()                 # arm alone; fmr6.Robot("table") for the table scene
    robot.set_q(fmr6.READY)
    print(robot.tcp())                   # tool centre point position (m) and rotation matrix
    q = robot.ik([0.75, 0.0, 0.55])      # joint angles that put the tool tip there (tool pointing down)
"""
from .robot import Robot, READY, JOINTS, ASSETS, model_path

__all__ = ["Robot", "READY", "JOINTS", "ASSETS", "model_path"]
__version__ = "0.1.0"
