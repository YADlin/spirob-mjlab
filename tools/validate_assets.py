#!/usr/bin/env python3
"""Validate that SpiRob mjlab package assets are present.

This does not require mjlab. It only checks the XML and mesh files that must
exist before MuJoCo/mjlab can compile the task.
"""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "src" / "spirob_mjlab" / "assets"
ROBOT_XML = ASSET_DIR / "spirob_robot.xml"

REQUIRED_XML = [
    ASSET_DIR / "spirob_robot.xml",
    ASSET_DIR / "spirob_egg.xml",
    ASSET_DIR / "spirob_pedestal.xml",
    ASSET_DIR / "spirob_bucket.xml",
]


def main() -> int:
    ok = True

    print(f"Package root: {ROOT}")
    print("\nChecking XML files:")
    for path in REQUIRED_XML:
        status = "OK" if path.exists() else "MISSING"
        print(f"  {status:7s} {path.relative_to(ROOT)}")
        ok &= path.exists()

    if not ROBOT_XML.exists():
        return 1

    tree = ET.parse(ROBOT_XML)
    mesh_files = [mesh.get("file") for mesh in tree.findall(".//mesh") if mesh.get("file")]

    print("\nChecking robot mesh files referenced by spirob_robot.xml:")
    missing = []
    for rel in mesh_files:
        mesh_path = ASSET_DIR / rel
        if mesh_path.exists():
            print(f"  OK      {mesh_path.relative_to(ROOT)}")
        else:
            print(f"  MISSING {mesh_path.relative_to(ROOT)}")
            missing.append(mesh_path)

    if missing:
        print("\nMissing mesh files. Copy link_001.stl ... link_021.stl into assets/meshes/.")
        ok = False

    print("\nResult:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
