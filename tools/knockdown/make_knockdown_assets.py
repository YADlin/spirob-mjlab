"""Derive the knockdown assets from the source files (reproducible; run from the repo root).

  assets/knockdown/source/spirob_robot_k2_2.148_20261024_0318.xml   (NYK, 24 Oct 2026; verbatim)
    -> assets/knockdown/spirob_robot.xml   remove floor plane + lights (mjlab supplies the scene,
                                           matching assets/spirob_robot.xml); everything else unchanged
  assets/spirob_egg.xml      -> assets/knockdown/egg.xml       colour only (green, for the camera)
  assets/spirob_pedestal.xml -> assets/knockdown/pedestal.xml  colour (blue) and shaft 20 mm shorter:
                                cup top 46 mm instead of 66 mm above the table, so the real drooping
                                robot (base top 88 mm, tip ~70 mm) meets the egg, not the pedestal.
The original egg/pedestal/robot assets are not modified.
"""

import re
from pathlib import Path

A = Path(__file__).resolve().parents[2] / "src" / "spirob_mjlab" / "assets"
K = A / "knockdown"
SOURCE = K / "source" / "spirob_robot_k2_2.148_20261024_0318.xml"

EGG_RGBA = "0.16 0.67 0.31 1"       # saturated green  (#29AB4F)
PEDESTAL_RGBA = "0.12 0.35 0.78 1"  # saturated blue   (#1F59C7)
PEDESTAL_DROP_M = 0.020


def robot():
    s = SOURCE.read_text()
    s, n_plane = re.subn(r'\n\s*<geom type="plane"[^>]*/>', "", s)
    s, n_light = re.subn(r"\n\s*<light [^>]*/>", "", s)
    assert (n_plane, n_light) == (1, 3), (n_plane, n_light)
    s = s.replace('<mujoco model="SpiRob">',
                  "<!-- Knockdown asset: derived from source/" + SOURCE.name + " by\n"
                  "     tools/knockdown/make_knockdown_assets.py. Removed: floor plane, 3 lights (scene items are\n"
                  "     provided by mjlab, as in assets/spirob_robot.xml). Robot bodies, joints, hulls, tendons and\n"
                  "     actuators are unchanged. -->\n<mujoco model=\"SpiRob_RobotOnly_Knockdown\">", 1)
    (K / "spirob_robot.xml").write_text(s)


def egg():
    s = (A / "spirob_egg.xml").read_text()
    s, n = re.subn(r'rgba="0.97 0.93 0.80 1.0"', f'rgba="{EGG_RGBA}"', s)
    assert n == 1
    s = s.replace('<mujoco model="SpiRob_Egg">', '<!-- Knockdown copy of ../spirob_egg.xml: colour changed only. -->\n<mujoco model="SpiRob_Egg_Knockdown">')
    (K / "egg.xml").write_text(s)


def pedestal():
    s = (A / "spirob_pedestal.xml").read_text()
    s, n = re.subn(r'rgba="0.25 0.25 0.25 1"', f'rgba="{PEDESTAL_RGBA}"', s)
    assert n == 1
    # shaft: centre 0.0285, half-length 0.03  -> centre 0.0185, half-length 0.02 (20 mm shorter)
    s, n1 = re.subn(r'name="ped_shaft" type="cylinder" pos="0 0 0.0285" size="0.008 0.03"',
                    'name="ped_shaft" type="cylinder" pos="0 0 0.0185" size="0.008 0.02"', s)
    # cup: centre 0.060 -> 0.040 (same 28 x 28 x 8 mm box)
    s, n2 = re.subn(r'name="ped_cup" type="box" pos="0 0 0.06"', 'name="ped_cup" type="box" pos="0 0 0.04"', s)
    assert (n1, n2) == (1, 1)
    s = s.replace('<mujoco model="SpiRob_Pedestal">',
                  '<!-- Knockdown copy of ../spirob_pedestal.xml: blue, and 20 mm lower (cup top 46 mm above the\n'
                  '     table when the pedestal root sits at z = 2 mm) so the drooping real robot meets the egg. -->\n'
                  '<mujoco model="SpiRob_Pedestal_Knockdown">')
    (K / "pedestal.xml").write_text(s)


if __name__ == "__main__":
    robot(); egg(); pedestal()
    print("wrote", *(p.name for p in sorted(K.glob("*.xml"))))
