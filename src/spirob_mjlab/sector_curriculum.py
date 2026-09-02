"""Shared geometry for the staged polar-sector curriculum.

The robot base is the polar origin and angles are measured counter-clockwise
from world +x.  Training progresses through three nested sectors; the final
sector spans 120--200 mm and 30--80 degrees.  All stages use the same reset
and evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


NOMINAL_EGG_XY_M = (0.05, 0.15)
NOMINAL_RADIUS_M = math.hypot(*NOMINAL_EGG_XY_M)
NOMINAL_ANGLE_DEG = math.degrees(
    math.atan2(NOMINAL_EGG_XY_M[1], NOMINAL_EGG_XY_M[0])
)

FULL_RADIUS_RANGE_M = (0.120, 0.200)
FULL_ANGLE_RANGE_DEG = (30.0, 80.0)

STAGE2_SECTOR_ARC1_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2Sector-Arc1"
STAGE2_SECTOR_ARC2_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2Sector-Arc2"
STAGE2_SECTOR_FULL_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2Sector-Full"


@dataclass(frozen=True)
class SectorSpec:
    """One nested acquisition sector in the shared curriculum."""

    task_id: str
    label: str
    expansion_fraction: float
    radius_range_m: tuple[float, float]
    angle_range_deg: tuple[float, float]


def interpolated_sector_bounds(
    expansion_fraction: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Interpolate both final bounds outward from the nominal position."""
    if not 0.0 < expansion_fraction <= 1.0:
        raise ValueError("expansion_fraction must be in (0, 1]")

    r_min = NOMINAL_RADIUS_M + expansion_fraction * (
        FULL_RADIUS_RANGE_M[0] - NOMINAL_RADIUS_M
    )
    r_max = NOMINAL_RADIUS_M + expansion_fraction * (
        FULL_RADIUS_RANGE_M[1] - NOMINAL_RADIUS_M
    )
    theta_min = NOMINAL_ANGLE_DEG + expansion_fraction * (
        FULL_ANGLE_RANGE_DEG[0] - NOMINAL_ANGLE_DEG
    )
    theta_max = NOMINAL_ANGLE_DEG + expansion_fraction * (
        FULL_ANGLE_RANGE_DEG[1] - NOMINAL_ANGLE_DEG
    )
    return (r_min, r_max), (theta_min, theta_max)


def _spec(task_id: str, label: str, expansion_fraction: float) -> SectorSpec:
    radius, angle = interpolated_sector_bounds(expansion_fraction)
    return SectorSpec(task_id, label, expansion_fraction, radius, angle)


SECTOR_SPECS = {
    STAGE2_SECTOR_ARC1_TASK_ID: _spec(
        STAGE2_SECTOR_ARC1_TASK_ID, "Arc-1", 1.0 / 3.0
    ),
    STAGE2_SECTOR_ARC2_TASK_ID: _spec(
        STAGE2_SECTOR_ARC2_TASK_ID, "Arc-2", 2.0 / 3.0
    ),
    STAGE2_SECTOR_FULL_TASK_ID: _spec(
        STAGE2_SECTOR_FULL_TASK_ID, "Full", 1.0
    ),
}


# Lower-left offsets, in millimetres relative to the nominal S1 spawn, for the
# 38 conservative two-millimetre cells supported by the archived S2-C map.
# These cells are rehearsal data, not the final hardware workspace boundary.
RETENTION_CELL_LOWER_LEFT_MM: tuple[tuple[float, float], ...] = tuple(
    (x_mm, y_mm)
    for y_mm, x_min_mm, x_max_mm in (
        (-10.0, -4.0, 2.0),
        (-8.0, -2.0, 6.0),
        (-6.0, -2.0, 8.0),
        (-4.0, 0.0, 8.0),
        (-2.0, 0.0, 8.0),
        (0.0, 0.0, 8.0),
        (2.0, 0.0, 8.0),
        (4.0, 0.0, 8.0),
        (6.0, 0.0, 8.0),
        (8.0, 4.0, 8.0),
    )
    for x_mm in (
        x_min_mm + 2.0 * index
        for index in range(round((x_max_mm - x_min_mm) / 2.0))
    )
)
RETENTION_CELL_SIZE_MM = 2.0


def polar_to_xy_m(radius_m: float, angle_deg: float) -> tuple[float, float]:
    """Convert robot-centred polar coordinates to world-local XY."""
    angle_rad = math.radians(angle_deg)
    return radius_m * math.cos(angle_rad), radius_m * math.sin(angle_rad)


def spawn_is_clear_xy_m(
    x_m: float,
    y_m: float,
    *,
    robot_centerline_x_m: float = 0.0,
    min_robot_centerline_clearance_m: float = 0.038,
    bucket_xy_m: tuple[float, float] = (-0.05, 0.15),
    min_bucket_center_clearance_m: float = 0.055,
) -> bool:
    """Pure-Python form of the conservative reset-time clearance screen."""
    robot_clear = (
        abs(x_m - robot_centerline_x_m) >= min_robot_centerline_clearance_m
    )
    bucket_clear = math.hypot(x_m - bucket_xy_m[0], y_m - bucket_xy_m[1]) >= (
        min_bucket_center_clearance_m
    )
    return robot_clear and bucket_clear
