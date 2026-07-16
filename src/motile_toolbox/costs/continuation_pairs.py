from __future__ import annotations

from typing import TYPE_CHECKING

from motile.variables import EdgeContinuation, Variable

if TYPE_CHECKING:
    from motile import Solver


class ContinuationPairs(Variable):
    """Binary variable for each pair of (incoming, outgoing) continuation edges
    at a node. A pair indicator is 1 if and only if both edges are active
    continuation edges (i.e., both selected and neither is a split/merge edge).

    This variable is useful for formulating costs that depend on consecutive
    edges in a track, such as penalizing changes in direction (curvature).
    """

    @staticmethod
    def instantiate(solver: Solver) -> list[tuple]:
        """Return keys for all (incoming, outgoing) edge pairs at each node."""
        return [
            (in_edge, out_edge)
            for node in solver.graph.nodes
            for in_edge in solver.graph.prev_edges[node]
            for out_edge in solver.graph.next_edges[node]
        ]

    @staticmethod
    def instantiate_constraints(solver: Solver):
        """Link pair indicators to EdgeContinuation variables."""
        cont_indicators = solver.get_variables(EdgeContinuation)
        pair_indicators = solver.get_variables(ContinuationPairs)

        for (in_edge, out_edge), pair_index in pair_indicators.items():
            c1 = cont_indicators[in_edge]
            c2 = cont_indicators[out_edge]

            # pair indicator = 1 <=> both edges are active continuations
            yield pair_index <= c1
            yield pair_index <= c2
            yield pair_index >= c1 + c2 - 1
