#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube
from grid_generator import HyperCubeBoundaryMarkers as BoundaryMarkers
from elastic_problem import LinearElasticProblem
from elastic_solver import DisplacementBCType
from elastic_solver import TractionBCType


class BlockTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "BlockTest"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = [(DisplacementBCType.fixed, BoundaryMarkers.left.value, None),
                     (DisplacementBCType.constant, BoundaryMarkers.right.value, (0.1, 0.0))]


def test_block_test():
    block_test = BlockTest(25)
    block_test.solve_problem()


if __name__ == "__main__":
    test_block_test()
