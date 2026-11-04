"""Writes the three Lab Kit notebooks (labs/lab-kit/). Edit the cells here, re-run, commit the .ipynb files.
    python tools/make_lab_kit.py"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "labs" / "lab-kit"
REPO = "https://github.com/kandapagari/fmr-robot-learning-labs"
COLAB = "https://colab.research.google.com/github/kandapagari/fmr-robot-learning-labs/blob/main/labs/lab-kit/"

SETUP = f'''# Setup: works in Google Colab and locally (after `pip install -e .` in the repo root)
import os, sys
if "google.colab" in sys.modules:
    os.environ["MUJOCO_GL"] = "egl"
    if not os.path.exists("fmr-robot-learning-labs"):
        !git clone -q {REPO}.git
    %cd fmr-robot-learning-labs
    !pip install -q -e .
import numpy as np, matplotlib.pyplot as plt, mediapy as media, mujoco
import fmr6


def show_video(frames, fps=30):
    """Play frames inline. Without ffmpeg (some local installs) show a strip of stills instead."""
    try:
        media.show_video(frames, fps=fps)
    except RuntimeError:
        media.show_images(frames[:: max(1, len(frames) // 5)], columns=5)'''

WIDE = '{"lookat": [0.45, 0.0, 0.45], "distance": 2.0, "azimuth": 140, "elevation": -18}'
NEAR = '{"lookat": [0.66, -0.05, 0.42], "distance": 1.2, "azimuth": 150, "elevation": -28}'

L1 = [
    ("md", f"""# Lab Kit L1 · Your robot lab in a browser tab

**Video:** Lab Kit L1 on [Foundation Models for Robotics](https://www.youtube.com/@FoundationModelsForRobotics)
**You need:** a Google account for Colab, or Python 3.10+ on your machine. No GPU. About 15 minutes.

Every lesson in the series comes with a lab like this one. This first one is a tour: load a robot, look inside it, move it, take pictures of it. Run the cells top to bottom with Shift+Enter."""),
    ("code", SETUP),
    ("md", """## 1 · Load a robot

`fmr6.Robot(scene, arm=...)` loads a MuJoCo model. The scene is `"arm"` (robot alone) or `"table"` (robot, table, cup and sponge). The arm is `"fr3"`, a Franka FR3 with the Franka Hand, or `"fmr6"`, the six-axis arm drawn in the videos."""),
    ("code", f"""robot = fmr6.Robot("table", arm="fr3")
media.show_image(robot.render(640, 400, camera={WIDE}))"""),
    ("md", """## 2 · Look inside

A robot model is a list of joints, each with limits, and a list of actuators (the motors that drive them). The FR3 has seven joints and one more actuator for the gripper."""),
    ("code", """for name, lo, hi in zip(robot.joints, np.degrees(robot.lo), np.degrees(robot.hi)):
    print(f"{name:12s} {lo:7.1f}° … {hi:6.1f}°")
print("actuators:", [robot.model.actuator(i).name for i in range(robot.model.nu)])
print("pose now (degrees):", np.degrees(robot.q()).round(1))"""),
    ("md", """## 3 · Move one joint

`robot.set_q(q)` places the arm at joint angles `q` (radians) and updates where every link is. Nothing is simulated yet: this is pure geometry, the same thing you'd do with a protractor on a drawing. Here we turn the base joint and then the elbow (joint 4)."""),
    ("code", f"""q = robot.ready.copy()
views = []
for j, deg in [(None, 0), (0, 40), (3, 35)]:
    if j is not None:
        q[j] += np.radians(deg)
    robot.set_q(q)
    views.append(robot.render(320, 240, camera={WIDE}))
media.show_images(views, titles=["ready", "joint 1 +40°", "then joint 4 +35°"])"""),
    ("md", """## 4 · Where is the hand?

The point that matters most is the **tool centre point** (TCP): the spot between the fingertips. `robot.tcp()` returns its position in metres and its orientation as a 3×3 rotation matrix. On the FR3 the hand points along the matrix's third column."""),
    ("code", """robot.set_q(robot.ready)
pos, R = robot.tcp()
print("TCP position (m):", pos.round(3))
print("hand points along:", R[:, robot.tool_axis].round(2), " (straight down is [0, 0, -1])")"""),
    ("md", """## 5 · Two arms, one table

FMR-6 has six joints and a simpler shape, which makes the maths in the videos easy to draw. The FR3 has seven, which gives it one spare degree of freedom: it can move its elbow while the hand stays still. Same table, same objects, same `Robot` methods."""),
    ("code", f"""pics = []
for arm in ("fr3", "fmr6"):
    r = fmr6.Robot("table", arm=arm)
    pics.append(r.render(400, 300, camera={WIDE}))
media.show_images(pics, titles=["Franka FR3 (7 joints)", "FMR-6 (6 joints)"])"""),
    ("md", """## 6 · A short animation

A video is a list of images. Sweep joint 1 back and forth and render a frame each time."""),
    ("code", f"""frames = []
for t in np.linspace(0, 2 * np.pi, 60):
    q = robot.ready.copy()
    q[0] = 0.6 * np.sin(t)
    robot.set_q(q)
    frames.append(robot.render(480, 300, camera={WIDE}))
show_video(frames, fps=30)"""),
    ("md", f"""## 7 · Try it

1. Find joint angles that put the fingertips just above the cup (hint: start from `robot.ready`, change one joint at a time, print `robot.tcp()[0]`). Then compare with the cup's position: `robot.data.body("cup").xpos`.
2. Set joint 4 to a value outside its limits and call `set_q`. What happens? (Nothing stops you: `set_q` is geometry only. Limits matter once we simulate physics in L3.)
3. Change `camera=` to look at the robot from above, then from the side.
4. Open the first real lab: [S0E1 · Small errors don't stay small]({COLAB.replace('lab-kit/', 's0e1-drift/s0e1_drift.ipynb')}).

**Next:** L2 drives the arm to a point you choose instead of to joint angles."""),
    ("code", "# your experiments here"),
]

L2 = [
    ("md", """# Lab Kit L2 · Drive the arm

**Video:** Lab Kit L2 on [Foundation Models for Robotics](https://www.youtube.com/@FoundationModelsForRobotics)
**You need:** Lab Kit L1. About 20 minutes.

In L1 you told the arm how to bend each joint. Usually you want the opposite: *put the hand there* and let the computer work out the joints. That's inverse kinematics (IK). Track A (A3, A4) explains how it works inside; here you learn to use it."""),
    ("code", SETUP),
    ("md", """## 1 · Put the hand at a point

`robot.ik(target)` returns joint angles that put the TCP at `target` (metres) with the hand pointing straight down. It starts from the current pose and improves it step by step."""),
    ("code", f"""robot = fmr6.Robot("table", arm="fr3")
target = np.array([0.60, 0.10, 0.45])
q = robot.ik(target)
pos, R = robot.tcp(q)
print("asked for", target, " got", pos.round(4), f" error {{np.linalg.norm(pos - target) * 1000:.2f}} mm")
media.show_image(robot.render(560, 360, camera={NEAR}))"""),
    ("md", """## 2 · Two ways to get from A to B

Moving from pose A to pose B, you can interpolate the **joint angles** (each joint turns at a steady rate) or the **hand position** (solve IK at every point along a straight line). Both arrive at B. The red line shows where the fingertips went."""),
    ("code", f"""A, B = np.array([0.55, -0.20, 0.45]), np.array([0.55, 0.20, 0.45])
qA = robot.ik(A, q0=robot.ready); qB = robot.ik(B, q0=qA)

def run(kind, n=40):
    tips, frames, q = [], [], qA
    for s in np.linspace(0, 1, n):
        q = qA + s * (qB - qA) if kind == "joint" else robot.ik(A + s * (B - A), q0=q)
        robot.set_q(q)
        tips.append(robot.tcp()[0])
        frames.append(robot.render(400, 280, camera={NEAR}, trail=np.array(tips), guide=np.array([A, B])))
    return np.array(tips), frames

tips_j, frames_j = run("joint")
tips_c, frames_c = run("cartesian")
print(f"joint interpolation: tip strays up to {{np.abs(tips_j[:, 0] - A[0]).max() * 1000:.0f}} mm off the straight line")
print(f"cartesian:           {{np.abs(tips_c[:, 0] - A[0]).max() * 1000:.1f}} mm")
media.show_images([frames_j[-1], frames_c[-1]], titles=["joint interpolation", "straight line (IK every step)"])"""),
    ("md", """Joint interpolation is simple and smooth for the motors, but the hand sweeps an arc. If something is in the way, or you're wiping a table, you want the straight line. Many learned policies output hand positions for this reason.

## 3 · Point the hand somewhere else

`tool_dir` sets where the hand points; `grip_dir` sets the direction the fingers close along. Both are directions in the world frame."""),
    ("code", f"""poses = [
    ("down", dict(tool_dir=(0, 0, -1))),
    ("down, fingers across y", dict(tool_dir=(0, 0, -1), grip_dir=(0, 1, 0))),
    ("tilted forward 45°", dict(tool_dir=(1, 0, -1))),
]
pics = []
for title, kw in poses:
    q = robot.ik([0.55, 0.0, 0.50], q0=robot.ready, **kw)
    pics.append(robot.render(320, 240, camera={NEAR}))
media.show_images(pics, titles=[p[0] for p in poses])"""),
    ("md", """## 4 · When IK can't do it

Ask for a point the arm can't reach and IK does its best, then stops with an error. Always check the error before you trust the joint angles."""),
    ("code", """for x in (0.60, 0.70, 0.75, 0.80, 0.90):
    t = np.array([x, 0.0, 0.45])
    q = robot.ik(t, q0=robot.ready)
    err = np.linalg.norm(robot.tcp(q)[0] - t)
    print(f"x = {x:.2f} m → error {err * 1000:7.1f} mm", "" if err < 1e-3 else "  ← out of reach")"""),
    ("md", """## 5 · Cameras and saving a video

`camera=` takes `lookat` (the point at the centre of the image), `distance`, `azimuth` (degrees around the vertical) and `elevation` (degrees up/down; negative looks down). Orbit the camera while the arm draws its straight line, then save an MP4 you can share."""),
    ("code", f"""orbit = []
for i, f in enumerate(frames_c + frames_c[::-1]):
    robot.set_q(robot.ik(tips_c[min(i, 2 * len(tips_c) - 1 - i)], q0=robot.q()))
    cam = {{"lookat": [0.55, 0.0, 0.45], "distance": 1.6, "azimuth": 120 + 1.5 * i, "elevation": -25}}
    orbit.append(robot.render(480, 320, camera=cam, trail=tips_c))
show_video(orbit, fps=30)
try:
    media.write_video("my_first_robot_video.mp4", orbit, fps=30)
    print("saved my_first_robot_video.mp4 (in Colab: Files panel on the left → download)")
except RuntimeError:
    print("no ffmpeg here, so no MP4; the frames above are still fine")"""),
    ("md", """## 6 · Try it

1. Make the hand draw a 20 cm square on the table, 5 cm above the surface (z = 0.43). Plot the four corners with `guide=`.
2. Repeat section 2 with `fmr6.Robot("table", arm="fmr6")`. Does joint interpolation stray more or less?
3. Find the furthest point straight ahead (y = 0, z = 0.45) the FR3 can reach with the hand pointing down, to the nearest centimetre.
4. The FR3 has seven joints and the target only fixes six numbers (3 position, 3 orientation). Start IK from two different `q0` and compare the joint angles you get. Same hand pose, different arm?

**Next:** L3 runs the real physics, so motors take time, objects fall and fingers can grip."""),
    ("code", "# your experiments here"),
]

L3 = [
    ("md", """# Lab Kit L3 · Make it yours

**Video:** Lab Kit L3 on [Foundation Models for Robotics](https://www.youtube.com/@FoundationModelsForRobotics)
**You need:** Lab Kit L2. About 30 minutes.

So far the arm teleported: `set_q` placed it, nothing pushed back. Now we switch on physics. Motors take time to get somewhere, objects have mass, and the fingers can squeeze a sponge. Then you'll add your own object to the scene and record a demonstration, the raw material for every learning lesson later in the series."""),
    ("code", SETUP),
    ("md", """## 0 · Running on your own machine

```bash
git clone https://github.com/kandapagari/fmr-robot-learning-labs.git
cd fmr-robot-learning-labs
pip install -e ".[dev]"
python -m mujoco.viewer --mjcf fmr6/assets/fr3/scene_table.xml   # interactive 3D viewer
```

In the viewer, double-click a body to select it, then Ctrl + right-drag to push it. Open the *Control* panel on the right to move each motor with a slider. (On macOS use `mjpython -m mujoco.viewer …`.)"""),
    ("md", """## 1 · Model and data

Under `fmr6.Robot` are two MuJoCo objects. `model` is everything that doesn't change: links, masses, joint limits, motors. `data` is everything that does: joint angles (`qpos`), velocities (`qvel`), motor commands (`ctrl`), time. `mujoco.mj_step(model, data)` advances the world by one timestep."""),
    ("code", """robot = fmr6.Robot("table", arm="fr3")
m, d = robot.model, robot.data
print("timestep:", m.opt.timestep, "s →", round(1 / m.opt.timestep), "steps per simulated second")
print("qpos has", m.nq, "numbers: 7 arm joints, 2 fingers, then 7 each for the cup and sponge (position + quaternion)")
print("ctrl has", m.nu, "numbers: 7 joint targets + gripper (0 closed … 255 open)")"""),
    ("md", """## 2 · Motors take time

Each FR3 joint has a position motor: set `d.ctrl[j]` to a target angle and the motor pushes toward it. Step the simulation and record how joint 4 follows a sudden 30° command."""),
    ("code", """mujoco.mj_resetDataKeyframe(m, d, 0)            # back to Franka's home pose, fingers open
d.ctrl[:7] = robot.ready; d.ctrl[7] = 255
target = robot.ready[3] + np.radians(30)
t, q4 = [], []
steps = int(1.0 / m.opt.timestep)              # 1 simulated second
for i in range(steps):
    if d.time >= 0.1:
        d.ctrl[3] = target                     # command a jump at t = 0.1 s
    mujoco.mj_step(m, d)
    t.append(d.time); q4.append(d.qpos[robot.qadr[3]])
plt.figure(figsize=(7, 3))
plt.plot(t, np.degrees(q4), label="joint 4 (actual)")
plt.axhline(np.degrees(target), ls="--", c="k", lw=0.8, label="command")
plt.xlabel("time (s)"); plt.ylabel("degrees"); plt.legend(); plt.tight_layout(); plt.show()"""),
    ("md", """The joint doesn't jump. It accelerates, overshoots slightly or not at all depending on the motor's stiffness and damping, and settles. A5 (control loops) is about exactly this curve.

## 3 · Pick up the sponge

A scripted pick: IK gives the joint targets, the motors move there over time, the gripper closes, the arm lifts. The sponge is 10 cm long and 6 cm wide, and the Franka Hand opens to 8 cm, so we grip it across its width with `grip_dir=(0, 1, 0)`."""),
    ("code", f"""def move(q_target, gripper, seconds, frames=None, log=None):
    \"\"\"Ramp the motor targets from where they are to q_target, stepping physics. gripper: 0 closed … 255 open.\"\"\"
    start, n = d.ctrl[:7].copy(), int(seconds / m.opt.timestep)
    log_every, frame_every = round(0.02 / m.opt.timestep), round((1 / 30) / m.opt.timestep)
    for i in range(n):
        s = min(1.0, (i + 1) / (0.7 * n)); s = s * s * (3 - 2 * s)    # smooth start and stop
        d.ctrl[:7] = start + s * (q_target - start); d.ctrl[7] = gripper
        mujoco.mj_step(m, d)
        if log is not None and i % log_every == 0:                    # 50 Hz, like a real robot log
            log.append(np.concatenate([d.qpos[robot.qadr], [d.ctrl[7]], d.body("sponge").xpos]))
        if frames is not None and i % frame_every == 0:                # ~30 fps video
            frames.append(robot.render(480, 320, camera={NEAR}))

mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)  # reset, then recompute positions from qpos
sponge = d.body("sponge").xpos.copy()
above = robot.ik(sponge + [0, 0, 0.08], q0=robot.ready, grip_dir=(0, 1, 0))
grasp = robot.ik(sponge + [0, 0, 0.005], q0=robot.ready, grip_dir=(0, 1, 0))
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)  # ik() moved the arm geometrically; undo that

frames, log = [], []
move(above, 255, 1.5, frames, log)     # over the sponge, fingers open
move(grasp, 255, 1.0, frames, log)     # down
move(grasp, 0, 0.8, frames, log)       # squeeze
move(above, 0, 1.5, frames, log)       # lift
print(f"sponge height: {{sponge[2]:.3f}} m → {{d.body('sponge').xpos[2]:.3f}} m")
show_video(frames, fps=30)"""),
    ("md", """## 4 · Add your own object

`mujoco.MjSpec` loads a scene as an editable description. Add a body, give it a free joint (so it can fall and slide) and a shape, compile, and hand the new model to `fmr6.Robot`."""),
    ("code", f"""spec = mujoco.MjSpec.from_file(str(fmr6.model_path("table", arm="fr3")))
block = spec.worldbody.add_body(name="block", pos=[0.60, 0.15, 0.55])     # 15 cm above the table
block.add_freejoint()
block.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.025], rgba=[0.2, 0.5, 0.9, 1], mass=0.1)
robot2 = fmr6.Robot(arm="fr3", model=spec.compile())
mujoco.mj_resetDataKeyframe(robot2.model, robot2.data, 0)
drop = []
for i in range(int(0.6 / robot2.model.opt.timestep)):                # 0.6 s: the block falls onto the table
    mujoco.mj_step(robot2.model, robot2.data)
    if i % 10 == 0:
        drop.append(robot2.render(400, 280, camera={NEAR}))
print("block rests at z =", robot2.data.body("block").xpos[2].round(3), "m (table top is at 0.38 m)")
show_video(drop, fps=30)"""),
    ("md", """## 5 · Save a demonstration

`log` from section 3 holds, 50 times a second, the joint angles, the gripper command and where the sponge was. That's a demonstration: what a learning robot is trained on. Save it; Season 2 trains a policy on files like this (in LeRobot's format, which A8 introduces)."""),
    ("code", """demo = np.array(log)
np.savez("demo_pick_sponge.npz", q=demo[:, :7], gripper=demo[:, 7], sponge=demo[:, 8:], hz=50)
print("saved", demo.shape[0], "steps =", demo.shape[0] / 50, "s of demonstration")
plt.figure(figsize=(7, 3))
plt.plot(np.arange(len(demo)) / 50, demo[:, 10], label="sponge height (m)")
plt.plot(np.arange(len(demo)) / 50, demo[:, 7] / 255 * 0.1 + 0.38, label="gripper (open ↑)")
plt.xlabel("time (s)"); plt.legend(); plt.tight_layout(); plt.show()"""),
    ("md", """## 6 · Try it

1. Pick up the blue block from section 4 instead of the sponge. (Build the pick with `robot2`.) Which way should the fingers close for a cube?
2. Try gripping the sponge along its length (`grip_dir=(1, 0, 0)`). What happens, and why?
3. Make the pick twice as fast. At what speed does the sponge slip out?
4. Move the cup with physics: push it 10 cm across the table with the closed fingers.
5. Record five picks with the sponge in slightly different places (`d.qpos` of its free joint) and save each to its own file. You've just made a tiny dataset.

That's the kit. Every lesson's lab from here on uses these same tools."""),
    ("code", "# your experiments here"),
]


def write(name, cells):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    stem = name.split("_")[0]
    for i, (kind, src) in enumerate(cells):
        c = {"cell_type": "markdown" if kind == "md" else "code", "id": f"{stem}-{i:02d}", "metadata": {},
             "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


write("l1_first_sim.ipynb", L1)
write("l2_drive_the_arm.ipynb", L2)
write("l3_make_it_yours.ipynb", L3)
print("wrote", OUT)
