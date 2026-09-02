"""Conservative workspace gate for the frozen S2-C controller.

This module contains no simulator or policy code.  It converts an egg/pedestal
XY offset relative to the nominal S1 spawn into an external dispatch decision.

The accepted set is the union of the 38 two-millimetre cells whose four corner
points each achieved at least 90 successes in 100 frozen-policy repetitions in
S2-C-workspace-map-v1.  None of these cells touches an engineering-invalid grid
point.  This is a controller-specific demonstrated region, not a claim that
locations outside it are physically unreachable.
"""

from __future__ import annotations

from enum import Enum
import math


WORKSPACE_EVIDENCE_ARCHIVE = "s2c_workspace_full_v1.tar.gz"
WORKSPACE_EVIDENCE_SHA256 = (
    "0f0b9fee0712161e48e51e71502ac9e5f906530fc3fa89bdd30f38fd9e3e9461"
)
WORKSPACE_PROTOCOL = "S2-C-workspace-map-v1"
WORKSPACE_CHECKPOINT_SHA256 = (
    "0dad0988485f39e623b59e791371ce80db58b8210aaee09eaf8d4b66d3d71a87"
)

GRID_MIN_MM = -10.0
GRID_MAX_MM = 10.0
CELL_SIZE_MM = 2.0

# For each y-cell from [-10, -8] through [8, 10] mm, give the closed x
# interval covered by the supported cells in that row.  Every row happens to
# contain one contiguous run, so this is clearer than embedding 38 cell IDs.
_SUPPORTED_X_INTERVALS_BY_Y_CELL: tuple[tuple[float, float], ...] = (
    (-4.0, 2.0),   # y in [-10, -8]
    (-2.0, 6.0),   # y in [ -8, -6]
    (-2.0, 8.0),   # y in [ -6, -4]
    (0.0, 8.0),    # y in [ -4, -2]
    (0.0, 8.0),    # y in [ -2,  0]
    (0.0, 8.0),    # y in [  0,  2]
    (0.0, 8.0),    # y in [  2,  4]
    (0.0, 8.0),    # y in [  4,  6]
    (0.0, 8.0),    # y in [  6,  8]
    (4.0, 8.0),    # y in [  8, 10]
)


class WorkspaceDecision(str, Enum):
    """External decision made before dispatching the S2-C policy."""

    ATTEMPT_MANIPULATION = "ATTEMPT_MANIPULATION"
    OUTSIDE_DEMONSTRATED_WORKSPACE = "OUTSIDE_DEMONSTRATED_WORKSPACE"
    NONFINITE_INPUT = "NONFINITE_INPUT"


def _candidate_y_cells(y_mm: float) -> tuple[int, ...]:
    """Return every closed grid cell containing ``y_mm``.

    A coordinate on a shared edge belongs to both neighbouring cells.  The
    gate accepts it when either cell belongs to the demonstrated union.
    """
    if y_mm < GRID_MIN_MM or y_mm > GRID_MAX_MM:
        return ()

    scaled = (y_mm - GRID_MIN_MM) / CELL_SIZE_MM
    nearest = round(scaled)
    on_edge = math.isclose(scaled, nearest, rel_tol=0.0, abs_tol=1e-10)

    if on_edge:
        return tuple(
            index
            for index in (nearest - 1, nearest)
            if 0 <= index < len(_SUPPORTED_X_INTERVALS_BY_Y_CELL)
        )

    index = math.floor(scaled)
    return (index,) if 0 <= index < len(_SUPPORTED_X_INTERVALS_BY_Y_CELL) else ()


def _inside_demonstrated_union(x_mm: float, y_mm: float) -> bool:
    for y_cell in _candidate_y_cells(y_mm):
        x_min, x_max = _SUPPORTED_X_INTERVALS_BY_Y_CELL[y_cell]
        if x_min <= x_mm <= x_max:
            return True
    return False


def stage2c_workspace_decision(
    offset_x_mm: float,
    offset_y_mm: float,
    *,
    localization_uncertainty_mm: float = 0.0,
) -> WorkspaceDecision:
    """Return the conservative external dispatch decision for S2-C.

    ``offset_x_mm`` and ``offset_y_mm`` are measured relative to the nominal
    egg/pedestal spawn used by S1.  ``localization_uncertainty_mm`` is a
    symmetric bound, not a standard deviation.  When it is positive, all four
    corners of the corresponding uncertainty box must remain inside the
    demonstrated union.

    This function is intentionally fail-closed: non-finite inputs are never
    passed to the controller.
    """
    values = (offset_x_mm, offset_y_mm, localization_uncertainty_mm)
    if not all(math.isfinite(value) for value in values):
        return WorkspaceDecision.NONFINITE_INPUT
    if localization_uncertainty_mm < 0.0:
        raise ValueError("localization_uncertainty_mm must be non-negative")

    radius = localization_uncertainty_mm
    corners = (
        (offset_x_mm - radius, offset_y_mm - radius),
        (offset_x_mm - radius, offset_y_mm + radius),
        (offset_x_mm + radius, offset_y_mm - radius),
        (offset_x_mm + radius, offset_y_mm + radius),
    )
    if all(_inside_demonstrated_union(x_mm, y_mm) for x_mm, y_mm in corners):
        return WorkspaceDecision.ATTEMPT_MANIPULATION
    return WorkspaceDecision.OUTSIDE_DEMONSTRATED_WORKSPACE
