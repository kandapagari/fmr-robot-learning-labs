import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "labs" / "s0e2-two-families"))
import fmr6  # noqa: E402
import two_families as tf  # noqa: E402


def setup_module():
    global robot, copier
    robot = fmr6.Robot("table", arm="fr3")
    X = tf.sample_targets(50, tf.LEFT_Y, seed=0)
    copier = tf.CopyDemos(k=8).fit(X, tf.demonstrate(robot, X))


def test_inside_the_demos_both_families_land():
    r = tf.compare(robot, copier, tf.sample_targets(30, tf.LEFT_Y, seed=1))
    assert r[:, 0].max() < 1e-3                       # IK: under a millimetre everywhere
    assert np.median(r[:, 1]) < 5e-3                  # copier: a few millimetres


def test_outside_the_demos_only_ik_lands():
    r = tf.compare(robot, copier, tf.sample_targets(30, tf.RIGHT_Y, seed=2))
    assert r[:, 0].max() < 1e-3
    assert np.median(r[:, 1]) > 2e-2                  # copier: centimetres off
    assert np.corrcoef(r[:, 2], r[:, 1])[0, 1] > 0.5  # and worse the further from the demos
