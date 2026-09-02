from __future__ import annotations

import math
import unittest

from spirob_mjlab.sector_curriculum import (
    FULL_ANGLE_RANGE_DEG,
    FULL_RADIUS_RANGE_M,
    NOMINAL_ANGLE_DEG,
    NOMINAL_EGG_XY_M,
    NOMINAL_RADIUS_M,
    RETENTION_CELL_LOWER_LEFT_MM,
    SECTOR_SPECS,
    STAGE2_SECTOR_ARC1_TASK_ID,
    STAGE2_SECTOR_ARC2_TASK_ID,
    STAGE2_SECTOR_FULL_TASK_ID,
    interpolated_sector_bounds,
    polar_to_xy_m,
    spawn_is_clear_xy_m,
)
from spirob_mjlab.workspace_gate import (
    WorkspaceDecision,
    stage2c_workspace_decision,
)


class SectorCurriculumTest(unittest.TestCase):
    def test_nominal_polar_coordinates_round_trip(self) -> None:
        x_m, y_m = polar_to_xy_m(NOMINAL_RADIUS_M, NOMINAL_ANGLE_DEG)
        self.assertAlmostEqual(x_m, NOMINAL_EGG_XY_M[0])
        self.assertAlmostEqual(y_m, NOMINAL_EGG_XY_M[1])

    def test_three_nested_specs_reach_the_full_sector(self) -> None:
        task_ids = (
            STAGE2_SECTOR_ARC1_TASK_ID,
            STAGE2_SECTOR_ARC2_TASK_ID,
            STAGE2_SECTOR_FULL_TASK_ID,
        )
        specs = [SECTOR_SPECS[task_id] for task_id in task_ids]
        self.assertEqual([spec.expansion_fraction for spec in specs], [1 / 3, 2 / 3, 1])
        self.assertGreater(specs[0].radius_range_m[0], specs[1].radius_range_m[0])
        self.assertGreater(specs[1].radius_range_m[0], specs[2].radius_range_m[0])
        self.assertLess(specs[0].radius_range_m[1], specs[1].radius_range_m[1])
        self.assertLess(specs[1].radius_range_m[1], specs[2].radius_range_m[1])
        self.assertEqual(specs[-1].radius_range_m, FULL_RADIUS_RANGE_M)
        self.assertEqual(specs[-1].angle_range_deg, FULL_ANGLE_RANGE_DEG)

    def test_interpolation_starts_from_nominal(self) -> None:
        radius, angle = interpolated_sector_bounds(1.0e-9)
        self.assertTrue(math.isclose(radius[0], NOMINAL_RADIUS_M, abs_tol=1.0e-9))
        self.assertTrue(math.isclose(radius[1], NOMINAL_RADIUS_M, abs_tol=1.0e-9))
        self.assertTrue(math.isclose(angle[0], NOMINAL_ANGLE_DEG, abs_tol=1.0e-7))
        self.assertTrue(math.isclose(angle[1], NOMINAL_ANGLE_DEG, abs_tol=1.0e-7))

    def test_retention_cells_match_the_archived_gate(self) -> None:
        self.assertEqual(len(RETENTION_CELL_LOWER_LEFT_MM), 38)
        for x_low_mm, y_low_mm in RETENTION_CELL_LOWER_LEFT_MM:
            decision = stage2c_workspace_decision(x_low_mm + 1.0, y_low_mm + 1.0)
            self.assertEqual(decision, WorkspaceDecision.ATTEMPT_MANIPULATION)

    def test_full_sector_contains_clear_and_collision_screened_points(self) -> None:
        clear_xy = polar_to_xy_m(0.20, 30.0)
        screened_xy = polar_to_xy_m(0.12, 80.0)
        self.assertTrue(spawn_is_clear_xy_m(*clear_xy))
        self.assertFalse(spawn_is_clear_xy_m(*screened_xy))

    def test_invalid_expansion_fraction_is_rejected(self) -> None:
        for fraction in (0.0, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    interpolated_sector_bounds(fraction)


if __name__ == "__main__":
    unittest.main()
