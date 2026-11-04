"""Build fmr6/assets/fr3/ from MuJoCo Menagerie: the Franka FR3 arm with the Franka Hand attached at the flange.

Menagerie ships the FR3 without a gripper, so we take the hand from its Panda model (same flange, same hand).
Both are Apache-2.0 (derived from Franka Robotics' franka_description). Re-run to update:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie /tmp/mjm
    (cd /tmp/mjm && git sparse-checkout set franka_fr3 franka_emika_panda)
    python tools/build_fr3.py /tmp/mjm
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mjm")
out = Path(__file__).resolve().parents[1] / "fmr6" / "assets" / "fr3"
commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

fr3 = (src / "franka_fr3" / "fr3.xml").read_text()
hand = (src / "franka_emika_panda" / "hand.xml").read_text()

# ——— hand: rename what would clash with the FR3 file (default classes and materials are global in MuJoCo) ———
for old, new in [('"visual"', '"hand_visual"'), ('"collision"', '"hand_collision"')]:
    hand = hand.replace(f"class={old}", f"class={new}")
for m in ("white", "off_white", "black"):
    hand = hand.replace(f'name="{m}"', f'name="hand_{m}"').replace(f'material="{m}"', f'material="hand_{m}"')
sec = hand[hand.index("<default>") + len("<default>"):hand.index("<asset>")]
defaults = sec[:sec.rindex("</default>")].strip()            # the whole <default class="panda"> tree
assets = re.search(r"<asset>(.*?)</asset>", hand, re.S).group(1)
body = re.search(r"<worldbody>(.*?)</worldbody>", hand, re.S).group(1).strip()
# mount at the flange the way Menagerie's panda.xml does, and add the Franka TCP (0.1034 m from the flange)
body = body.replace('<body name="hand" childclass="panda" quat="0 0 0 1">',
                    '<body name="hand" childclass="panda" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">\n'
                    '      <site name="tcp" pos="0 0 0.1034" size="0.005" rgba="1 0.3 0.1 0.8" group="4"/>', 1)
rest = {k: re.search(rf"<{k}>(.*?)</{k}>", hand, re.S).group(1) for k in ("contact", "tendon", "equality", "actuator")}
rest["actuator"] = rest["actuator"].replace('name="actuator8"', 'name="gripper"')

# ——— splice into the FR3 file ———
x = fr3.replace('<mujoco model="fr3">', f'<!-- Franka FR3 + Franka Hand, assembled by tools/build_fr3.py from MuJoCo Menagerie {commit[:12]} (Apache-2.0). -->\n<mujoco model="fr3_hand">')
x = x.replace('meshdir="assets"', 'meshdir="meshes" autolimits="true"')
cut = x[:x.index("<asset>")].rindex("</default>")                # close of the outer <default>
x = x[:cut] + "  " + defaults + "\n  " + x[cut:]
x = x.replace("</asset>", assets.rstrip() + "\n  </asset>", 1)
x = x.replace('<site name="attachment_site" pos="0 0 0.107"/>', '<site name="attachment_site" pos="0 0 0.107"/>\n' + body, 1)
x = x.replace("</contact>", rest["contact"].rstrip() + "\n  </contact>", 1)
x = x.replace("</actuator>", rest["actuator"].rstrip() + "\n  </actuator>\n\n  <tendon>" + rest["tendon"] + "</tendon>\n\n  <equality>" + rest["equality"] + "</equality>", 1)
# keyframe: 7 arm joints + 2 fingers (open), ctrl: 7 arm + gripper (255 = open)
x = re.sub(r'<key name="home" qpos="([^"]+)" ctrl="([^"]+)"/>',
           lambda m: f'<key name="home" qpos="{m.group(1)} 0.04 0.04" ctrl="{m.group(2)} 255"/>', x)

out.mkdir(parents=True, exist_ok=True)
(out / "meshes").mkdir(exist_ok=True)
(out / "fr3_hand.xml").write_text(x)
for f in sorted(set(re.findall(r'file="([^"]+)"', x))):
    for d in (src / "franka_fr3" / "assets", src / "franka_emika_panda" / "assets"):
        if (d / f).exists():
            shutil.copy2(d / f, out / "meshes" / f)
            break
    else:
        raise FileNotFoundError(f)
shutil.copy2(src / "franka_fr3" / "LICENSE", out / "LICENSE")
print("wrote", out, "from menagerie", commit[:12])
