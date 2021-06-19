#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import PointSubDomain
from grid_generator import hyper_cube
from grid_generator import HyperCubeBoundaryMarkers
from elastic_problem import ElasticProblem
from elastic_solver import DisplacementBCType
from elastic_law import NeoHooke, NeoHookeIncompressible
import dolfin as dlfn


class TensileTest(ElasticProblem):
    def __init__(self, n_points, elastic_law, main_dir=None, bc_type="floating", polynomial_degree=2):
        super().__init__(elastic_law, main_dir, polynomial_degree=polynomial_degree)

        assert isinstance(n_points, int)
        assert n_points > 0
        self._n_points = n_points

        assert isinstance(bc_type, str)
        assert bc_type in ("floating", "clamped", "clamped_free", "pointwise")
        self._bc_type = bc_type

        if self._bc_type == "floating":
            self._problem_name = "TensileTest"
        elif self._bc_type == "clamped":
            self._problem_name = "TensileTestClamped"
        elif self._bc_type == "clamped_free":
            self._problem_name = "TensileTestClampedFree"
        elif self._bc_type == "pointwise":
            self._problem_name = "TensileTestPointwise"

        # self.set_parameters(C=1.e4, D=1.)
        self.set_parameters(E=210., nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        self._bcs = []
        BCType = DisplacementBCType
        if self._bc_type == "floating":
            self._bcs.append((BCType.fixed_component, HyperCubeBoundaryMarkers.left.value, 0, None))
            self._bcs.append((BCType.fixed_component, HyperCubeBoundaryMarkers.bottom.value, 1, None))
            self._bcs.append((BCType.constant_component, HyperCubeBoundaryMarkers.right.value, 0, 0.1))
        elif self._bc_type == "clamped":
            self._bcs.append((BCType.fixed, HyperCubeBoundaryMarkers.left.value, None))
            self._bcs.append((BCType.constant, HyperCubeBoundaryMarkers.right.value, (0.1, 0.0)))
        elif self._bc_type == "clamped_free":
            self._bcs.append((BCType.fixed, HyperCubeBoundaryMarkers.left.value, None))
            self._bcs.append((BCType.constant_component, HyperCubeBoundaryMarkers.right.value, 0, 0.1))
        elif self._bc_type == "pointwise":
            gamma01 = PointSubDomain((0.0, ) * self._space_dim, tol=1e-10)
            gamma02 = dlfn.CompiledSubDomain("near(x[0], 0.0)")
            self._bcs.append((BCType.fixed_pointwise, gamma01, None))
            self._bcs.append((BCType.fixed_component_pointwise, gamma02, 0, None))
            self._bcs.append((BCType.constant_component, HyperCubeBoundaryMarkers.right.value, 0, 0.1))

    def postprocess_solution(self):
        # compute stresses
        stress_tensor = self._compute_stress_tensor()
        # add stress components to the field output
        component_indices = []
        for i in range(self.space_dim):
            for j in range(i, self.space_dim):
                component_indices.append((i + 1, j + 1))
        for k, stress in enumerate(stress_tensor.split()):
            stress.rename("S{0}{1}".format(*component_indices[k]), "")
            self._add_to_field_output(stress)
        self._add_to_field_output(self._compute_volume_ratio())
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


def test_tensile_test():
    for elastic_law in [NeoHooke(), NeoHookeIncompressible()]:
        for bc_type in ("floating", "clamped", "clamped_free", "pointwise"):
            tensile_test = TensileTest(25, elastic_law, bc_type=bc_type)
            print(f"Running {tensile_test._problem_name} with {bc_type} boundary condition type.")
            tensile_test.solve_problem()
            print()


if __name__ == "__main__":
    test_tensile_test()
