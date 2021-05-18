#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube
from grid_generator import HyperCubeBoundaryMarkers as BoundaryMarkers
from elastic_problem import LinearElasticProblem
from elastic_solver import DisplacementBCType
from elastic_solver import TractionBCType
import dolfin as dlfn


class ClampedBarTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "ClampedBarTest"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = [(DisplacementBCType.fixed, BoundaryMarkers.left.value, None),
                     (DisplacementBCType.constant, BoundaryMarkers.right.value, (0.1, 0.0))]

    def postprocess_solution(self):
        # compute stresses
        stress_tensor = self._compute_stress_tensor()
        # add stress components to the field output
        component_indices = []
        for i in range(self.space_dim):
            for j in range(i, self.space_dim):
                component_indices.append((i+1, j+1))
        for k, stress in enumerate(stress_tensor.split()):
            stress.rename("S{0}{1}".format(*component_indices[k]), "")
            self._add_to_field_output(stress)


class TensileTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "TensileTest"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        # gamma = dlfn.CompiledSubDomain("near(x[0], 0.0) && near(x[1], 0.0) && on_boundary")
        self._bcs = [(DisplacementBCType.fixed_component, BoundaryMarkers.left.value, 0, None),
                     (DisplacementBCType.fixed_component, BoundaryMarkers.bottom.value, 1, None),
                     (DisplacementBCType.constant_component, BoundaryMarkers.right.value, 0, 0.1)]

    def postprocess_solution(self):
        # compute stresses
        stress_tensor = self._compute_stress_tensor()
        # add stress components to the field output
        component_indices = []
        for i in range(self.space_dim):
            for j in range(i, self.space_dim):
                component_indices.append((i+1, j+1))
        for k, stress in enumerate(stress_tensor.split()):
            stress.rename("S{0}{1}".format(*component_indices[k]), "")
            self._add_to_field_output(stress)
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i,j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


class DisplacementControlledShearTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "DisplacementControlledShearTest"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        # gamma = dlfn.CompiledSubDomain("near(x[0], 0.0) && near(x[1], 0.0) && on_boundary")
        self._bcs = [(DisplacementBCType.fixed, BoundaryMarkers.bottom.value, None),
                     (DisplacementBCType.constant, BoundaryMarkers.top.value, (0.1, 0.0))]

    def postprocess_solution(self):
        # compute stresses
        stress_tensor = self._compute_stress_tensor()
        # add stress components to the field output
        component_indices = []
        for i in range(self.space_dim):
            for j in range(i, self.space_dim):
                component_indices.append((i+1, j+1))
        for k, stress in enumerate(stress_tensor.split()):
            stress.rename("S{0}{1}".format(*component_indices[k]), "")
            self._add_to_field_output(stress)
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i,j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


class TractionControlledShearTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "TractionControlledShearTest"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        # gamma = dlfn.CompiledSubDomain("near(x[0], 0.0) && near(x[1], 0.0) && on_boundary")
        self._bcs = [(DisplacementBCType.fixed, BoundaryMarkers.bottom.value, None),
                     (DisplacementBCType.fixed_component, BoundaryMarkers.top.value, 1, None),
                     (TractionBCType.constant_component, BoundaryMarkers.top.value, 0, 0.1)]

    def postprocess_solution(self):
        # compute stresses
        stress_tensor = self._compute_stress_tensor()
        # add stress components to the field output
        component_indices = []
        for i in range(self.space_dim):
            for j in range(i, self.space_dim):
                component_indices.append((i+1, j+1))
        for k, stress in enumerate(stress_tensor.split()):
            stress.rename("S{0}{1}".format(*component_indices[k]), "")
            self._add_to_field_output(stress)
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i,j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


class BodyForceTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "BodyForceTest"

        self.set_parameters(E=210.0, nu=0.3, lref=1.0, bref=25.0)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_body_force(self):
        self._body_force = dlfn.Expression(
                ("x[0]*x[0] * (1.0 - x[0]*x[0]) * x[1]",
                 "x[1]*x[1] * (1.0 - x[1]*x[1]) * x[0]"), degree=2)

    def set_boundary_conditions(self):
        # boundary conditions
        self._bcs = [(DisplacementBCType.fixed, BoundaryMarkers.left.value, None),
                     (DisplacementBCType.fixed, BoundaryMarkers.right.value, None),
                     (DisplacementBCType.fixed, BoundaryMarkers.bottom.value, None),
                     (DisplacementBCType.fixed, BoundaryMarkers.top.value, None)]

    def postprocess_solution(self):
        # compute stresses
        stress_tensor = self._compute_stress_tensor()
        # add stress components to the field output
        component_indices = []
        for i in range(self.space_dim):
            for j in range(i, self.space_dim):
                component_indices.append((i+1, j+1))
        for k, stress in enumerate(stress_tensor.split()):
            stress.rename("S{0}{1}".format(*component_indices[k]), "")
            self._add_to_field_output(stress)


class BCFunctionTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "BCFunctionTest"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        fun01 = dlfn.Expression(("x[1]*x[1] * (1.0 - x[1]*x[1])", "0.0"), degree=2)
        fun02 = dlfn.Expression("x[0]*x[0] * (1.0 - x[0]*x[0])", degree=2)
        self._bcs = [(DisplacementBCType.fixed, BoundaryMarkers.left.value, None),
                     (DisplacementBCType.function, BoundaryMarkers.right.value, fun01),
                     (DisplacementBCType.function_component, BoundaryMarkers.top.value, 1, fun02)]

    def postprocess_solution(self):
        # compute stresses
        stress_tensor = self._compute_stress_tensor()
        # add stress components to the field output
        component_indices = []
        for i in range(self.space_dim):
            for j in range(i, self.space_dim):
                component_indices.append((i+1, j+1))
        for k, stress in enumerate(stress_tensor.split()):
            stress.rename("S{0}{1}".format(*component_indices[k]), "")
            self._add_to_field_output(stress)


def test_block_test():
    block_test = ClampedBarTest(25)
    block_test.solve_problem()


def test_tensile_test():
    tensile_test = TensileTest(25)
    tensile_test.solve_problem()


def test_shear_test():
    shear_test_disp = DisplacementControlledShearTest(25)
    shear_test_disp.solve_problem()
    shear_test_traction = TractionControlledShearTest(25)
    shear_test_traction.solve_problem()


def test_body_force():
    body_force_test = BodyForceTest(25)
    body_force_test.solve_problem()


def test_bc_function():
    bc_function_test = BCFunctionTest(25)
    bc_function_test.solve_problem()


if __name__ == "__main__":
    test_block_test()
    test_tensile_test()
    test_shear_test()
    test_body_force()
    test_bc_function()