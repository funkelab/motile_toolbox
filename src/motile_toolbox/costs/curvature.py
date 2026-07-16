from __future__ import annotations

from typing import TYPE_CHECKING

import motile.costs

from .continuation_pairs import ContinuationPairs

if TYPE_CHECKING:
    from motile import Solver
    from motile._types import EdgeId


class CurvatureCost(motile.costs.Cost):
    """Cost that penalizes changes in direction along a track.

    For each pair of consecutive continuation edges, this cost computes the
    difference in displacement (a proxy for curvature) and adds it as a cost.
    Higher curvature means higher cost.

    Args:
        position_attribute:
            The name of the node attribute holding the spatial position.
        weight:
            A scalar weight for the cost.
    """

    def __init__(self, position_attribute: str, weight: float = 1.0):
        self.position_attribute = position_attribute
        self.weight = motile.costs.Weight(weight)

    def apply(self, solver: Solver) -> None:
        """Add curvature costs to each continuation-edge pair."""
        pair_indicators = solver.get_variables(ContinuationPairs)

        for (in_edge, out_edge), index in pair_indicators.items():
            in_offset = self._get_edge_offset(solver, in_edge)
            out_offset = self._get_edge_offset(solver, out_edge)

            curvature_cost = abs(out_offset - in_offset)

            solver.add_variable_cost(index, curvature_cost, self.weight)

    def _get_edge_offset(self, solver: Solver, edge: EdgeId) -> float:
        pos_v = solver.graph.nodes[edge[1]][self.position_attribute]
        pos_u = solver.graph.nodes[edge[0]][self.position_attribute]
        return pos_v - pos_u
