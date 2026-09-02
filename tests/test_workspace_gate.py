from __future__ import annotations

import math
import unittest

from spirob_mjlab.workspace_gate import (
    WorkspaceDecision,
    stage2c_workspace_decision,
)


# Cell indices use the workspace-map convention: index 0 is [-10, -8] mm.
SUPPORTED_CELLS = {
    (3, 0), (4, 0), (5, 0),
    (4, 1), (5, 1), (6, 1), (7, 1),
    (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
    (5, 3), (6, 3), (7, 3), (8, 3),
    (5, 4), (6, 4), (7, 4), (8, 4),
    (5, 5), (6, 5), (7, 5), (8, 5),
    (5, 6), (6, 6), (7, 6), (8, 6),
    (5, 7), (6, 7), (7, 7), (8, 7),
    (5, 8), (6, 8), (7, 8), (8, 8),
    (7, 9), (8, 9),
}


class Stage2CWorkspaceGateTest(unittest.TestCase):
    def test_every_cell_center_matches_archived_mask(self) -> None:
        for y_cell in range(10):
            for x_cell in range(10):
                x_mm = -9.0 + 2.0 * x_cell
                y_mm = -9.0 + 2.0 * y_cell
                expected = (
                    WorkspaceDecision.ATTEMPT_MANIPULATION
                    if (x_cell, y_cell) in SUPPORTED_CELLS
                    else WorkspaceDecision.OUTSIDE_DEMONSTRATED_WORKSPACE
                )
                with self.subTest(x_cell=x_cell, y_cell=y_cell):
                    self.assertEqual(stage2c_workspace_decision(x_mm, y_mm), expected)

    def test_nominal_and_closed_supported_edges_are_accepted(self) -> None:
        self.assertEqual(
            stage2c_workspace_decision(0.0, 0.0),
            WorkspaceDecision.ATTEMPT_MANIPULATION,
        )
        self.assertEqual(
            stage2c_workspace_decision(8.0, 0.0),
            WorkspaceDecision.ATTEMPT_MANIPULATION,
        )
        self.assertEqual(
            stage2c_workspace_decision(6.0, 10.0),
            WorkspaceDecision.ATTEMPT_MANIPULATION,
        )

    def test_known_engineering_invalid_boundary_is_rejected(self) -> None:
        for y_mm in (0.0, 6.0):
            with self.subTest(y_mm=y_mm):
                self.assertEqual(
                    stage2c_workspace_decision(10.0, y_mm),
                    WorkspaceDecision.OUTSIDE_DEMONSTRATED_WORKSPACE,
                )

    def test_uncertainty_box_must_fit_completely(self) -> None:
        self.assertEqual(
            stage2c_workspace_decision(
                8.0,
                0.0,
                localization_uncertainty_mm=0.1,
            ),
            WorkspaceDecision.OUTSIDE_DEMONSTRATED_WORKSPACE,
        )
        self.assertEqual(
            stage2c_workspace_decision(
                4.0,
                0.0,
                localization_uncertainty_mm=0.1,
            ),
            WorkspaceDecision.ATTEMPT_MANIPULATION,
        )

    def test_nonfinite_input_fails_closed(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertEqual(
                    stage2c_workspace_decision(value, 0.0),
                    WorkspaceDecision.NONFINITE_INPUT,
                )

    def test_negative_uncertainty_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            stage2c_workspace_decision(
                0.0,
                0.0,
                localization_uncertainty_mm=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
