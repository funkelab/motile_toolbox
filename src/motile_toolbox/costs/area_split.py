from __future__ import annotations

from typing import TYPE_CHECKING

from motile.costs import Cost, Weight
from motile.variables import EdgeSplitPair

if TYPE_CHECKING:
    from motile import Solver


class AreaSplitCost(Cost):
    """Cost for :class:`~motile.variables.EdgeSplitPair` variables based on
    area conservation.

    For each pair of edges forming a split, the cost is the absolute difference
    between the parent's area and the sum of the two children's areas. A
    division that perfectly conserves area has zero cost.

    Args:
        area_attribute:
            The name of the node attribute holding the area (or volume).

        weight:
            The weight to apply to the area difference. Default is ``1.0``.

        constant:
            A constant cost for each active split pair. Default is ``0.0``.
    """

    def __init__(
        self,
        area_attribute: str,
        weight: float = 1.0,
        constant: float = 0.0,
    ) -> None:
        self.area_attribute = area_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)

    def apply(self, solver: Solver) -> None:
        """Add area-conservation costs to each split-edge pair."""
        pair_variables = solver.get_variables(EdgeSplitPair)

        for (e1, e2), index in pair_variables.items():
            parent = e1[0]  # both edges share the same source
            child1 = e1[1]
            child2 = e2[1]

            area_parent = solver.graph.nodes[parent][self.area_attribute]
            area_child1 = solver.graph.nodes[child1][self.area_attribute]
            area_child2 = solver.graph.nodes[child2][self.area_attribute]

            area_diff = abs(area_parent - (area_child1 + area_child2))

            solver.add_variable_cost(index, area_diff, self.weight)
            solver.add_variable_cost(index, 1.0, self.constant)
