#!/usr/bin/env python3
"""Extract the robot entity XML from the user's full SpiRob scene XML.

Usage:
  python tools/prepare_from_full_xml.py /path/to/spirob.xml

The full XML contains robot, egg, pedestal, bucket, and floor. mjlab wants each
free/detached object as a separate EntityCfg, so this keeps only `robot_base` in
spirob_robot.xml and leaves egg/pedestal/bucket in the provided standalone XMLs.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "src" / "spirob_mjlab" / "assets"
OUT_XML = ASSET_DIR / "spirob_robot.xml"


def indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    full_xml = Path(sys.argv[1]).expanduser().resolve()
    if not full_xml.exists():
        print(f"Missing XML: {full_xml}")
        return 1

    tree = ET.parse(full_xml)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("No <worldbody> in full XML")

    robot_base = None
    for body in worldbody.findall("body"):
        if body.get("name") == "robot_base":
            robot_base = deepcopy(body)
            break
    if robot_base is None:
        raise RuntimeError("No <body name='robot_base'> found")

    new_root = ET.Element(root.tag, root.attrib)
    for tag in ("compiler", "option", "visual", "default", "asset"):
        elem = root.find(tag)
        if elem is not None:
            new_root.append(deepcopy(elem))

    new_world = ET.SubElement(new_root, "worldbody")
    # Keep lights if present. Terrain is supplied by mjlab, not this robot entity.
    for child in worldbody:
        if child.tag == "light":
            new_world.append(deepcopy(child))
    new_world.append(robot_base)

    for tag in ("tendon", "actuator", "sensor"):
        elem = root.find(tag)
        if elem is not None:
            new_root.append(deepcopy(elem))

    indent(new_root)
    OUT_XML.write_text("<?xml version='1.0' encoding='utf-8'?>\n" + ET.tostring(new_root, encoding="unicode"))
    print(f"Wrote {OUT_XML.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
