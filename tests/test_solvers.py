#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import PointSubDomain
from grid_generator import hyper_cube
from grid_generator import HyperCubeBoundaryMarkers as BoundaryMarkers
from elastic_problem import LinearElasticProblem, NonlinearElasticProblem
from elastic_solver import DisplacementBCType
from elastic_solver import TractionBCType
import dolfin as dlfn


class TensileTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None, bc_type="floating"):
        super().__init__(main_dir)
        
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

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        self._bcs = []
        BCType = DisplacementBCType
        if self._bc_type == "floating":
            self._bcs.append((BCType.fixed_component, BoundaryMarkers.left.value, 0, None))
            self._bcs.append((BCType.fixed_component, BoundaryMarkers.bottom.value, 1, None))
            self._bcs.append((BCType.constant_component, BoundaryMarkers.right.value, 0, 0.1))
        elif self._bc_type == "clamped":
            self._bcs.append((BCType.fixed, BoundaryMarkers.left.value, None))
            self._bcs.append((BCType.constant, BoundaryMarkers.right.value, (0.1, 0.0)))
        elif self._bc_type == "clamped_free":
            self._bcs.append((BCType.fixed, BoundaryMarkers.left.value, None))
            self._bcs.append((BCType.constant_component, BoundaryMarkers.right.value, 0, 0.1))
        elif self._bc_type == "pointwise":
            gamma01 = PointSubDomain((0.0, ) * self._space_dim, tol=1e-10)
            gamma02 = dlfn.CompiledSubDomain("near(x[0], 0.0)")
            self._bcs.append((BCType.fixed_pointwise, gamma01, None))
            self._bcs.append((BCType.fixed_component_pointwise, gamma02, 0, None))
            self._bcs.append((BCType.constant_component, BoundaryMarkers.right.value, 0, 0.1))

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


class ShearTest(LinearElasticProblem):
    def __init__(self, n_points, main_dir=None, bc_type="displacement"):
        super().__init__(main_dir)

        assert isinstance(n_points, int)
        assert n_points > 0
        self._n_points = n_points
        
        assert isinstance(bc_type, str)
        assert bc_type in ("displacement", "traction")
        self._bc_type = bc_type

        if self._bc_type == "displacement":
            self._problem_name = "DisplacementControlledShearTest"
        elif self._bc_type == "traction":
            self._problem_name = "TractionControlledShearTest"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        self._bcs = []
        self._bcs.append((DisplacementBCType.fixed, BoundaryMarkers.bottom.value, None)),

        if self._bc_type == "displacement":
            self._bcs.append((DisplacementBCType.constant, BoundaryMarkers.top.value, (0.1, 0.0)))

        elif self._bc_type == "traction":
            self._bcs.append((DisplacementBCType.fixed_component, BoundaryMarkers.top.value, 1, None))
            self._bcs.append((TractionBCType.constant_component, BoundaryMarkers.top.value, 0, 0.1))

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


def test_tensile_test():
    for bc_type in ("floating", "clamped", "clamped_free", "pointwise"):
        tensile_test = TensileTest(25, bc_type=bc_type)
        print(f"Running {tensile_test._problem_name} with {bc_type} boundary condition type.")
        tensile_test .solve_problem()
        print()


def test_shear_test():
    for bc_type in ("displacement", "traction"):
        shear_test = ShearTest(25, bc_type=bc_type)
        print(f"Running {shear_test._problem_name} with {bc_type} boundary condition type.")
        shear_test.solve_problem()
        print()

def test_body_force():
    body_force_test = BodyForceTest(25)
    print(f"Running {body_force_test._problem_name}.")
    body_force_test.solve_problem()
    print()


def test_bc_function():
    bc_function_test = BCFunctionTest(25)
    print(f"Running {bc_function_test._problem_name}.")
    bc_function_test.solve_problem()
    print()


class NonlinearTensileTest(NonlinearElasticProblem):
    def __init__(self, n_points, main_dir=None, bc_type="floating"):
        super().__init__(main_dir)
        
        assert isinstance(n_points, int)
        assert n_points > 0
        self._n_points = n_points
        
        assert isinstance(bc_type, str)
        assert bc_type in ("floating", "clamped", "clamped_free", "pointwise")
        self._bc_type = bc_type

        if self._bc_type == "floating":
            self._problem_name = "NonlinearTensileTest"
        elif self._bc_type == "clamped":
            self._problem_name = "NonlinearTensileTestClamped"
        elif self._bc_type == "clamped_free":
            self._problem_name = "NonlinearTensileTestClampedFree"
        elif self._bc_type == "pointwise":
            self._problem_name = "NonlinearTensileTestPointwise"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        self._bcs = []
        BCType = DisplacementBCType
        if self._bc_type == "floating":
            self._bcs.append((BCType.fixed_component, BoundaryMarkers.left.value, 0, None))
            self._bcs.append((BCType.fixed_component, BoundaryMarkers.bottom.value, 1, None))
            self._bcs.append((BCType.constant_component, BoundaryMarkers.right.value, 0, 0.1))
        elif self._bc_type == "clamped":
            self._bcs.append((BCType.fixed, BoundaryMarkers.left.value, None))
            self._bcs.append((BCType.constant, BoundaryMarkers.right.value, (0.1, 0.0)))
        elif self._bc_type == "clamped_free":
            self._bcs.append((BCType.fixed, BoundaryMarkers.left.value, None))
            self._bcs.append((BCType.constant_component, BoundaryMarkers.right.value, 0, 0.1))
        elif self._bc_type == "pointwise":
            gamma01 = PointSubDomain((0.0, ) * self._space_dim, tol=1e-10)
            gamma02 = dlfn.CompiledSubDomain("near(x[0], 0.0)")
            self._bcs.append((BCType.fixed_pointwise, gamma01, None))
            self._bcs.append((BCType.fixed_component_pointwise, gamma02, 0, None))
            self._bcs.append((BCType.constant_component, BoundaryMarkers.right.value, 0, 0.1))

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


class NonlinearShearTest(NonlinearElasticProblem):
    def __init__(self, n_points, main_dir=None, bc_type="displacement"):
        super().__init__(main_dir)

        assert isinstance(n_points, int)
        assert n_points > 0
        self._n_points = n_points
        
        assert isinstance(bc_type, str)
        assert bc_type in ("displacement", "traction")
        self._bc_type = bc_type

        if self._bc_type == "displacement":
            self._problem_name = "NonlinearDisplacementControlledShearTest"
        elif self._bc_type == "traction":
            self._problem_name = "NonlinearTractionControlledShearTest"

        self.set_parameters(E=210.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        self._bcs = []
        self._bcs.append((DisplacementBCType.fixed, BoundaryMarkers.bottom.value, None)),

        if self._bc_type == "displacement":
            self._bcs.append((DisplacementBCType.constant, BoundaryMarkers.top.value, (0.1, 0.0)))

        elif self._bc_type == "traction":
            self._bcs.append((DisplacementBCType.fixed_component, BoundaryMarkers.top.value, 1, None))
            self._bcs.append((TractionBCType.constant_component, BoundaryMarkers.top.value, 0, 0.1))

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


class NonlinearBodyForceTest(NonlinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "NonlinearBodyForceTest"

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


class NonlinearBCFunctionTest(NonlinearElasticProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "NonlinearBCFunctionTest"

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


def test_nonlinear_tensile_test():
    for bc_type in ("floating", "clamped", "clamped_free", "pointwise"):
        tensile_test = NonlinearTensileTest(25, bc_type=bc_type)
        print(f"Running {tensile_test._problem_name} with {bc_type} boundary condition type.")
        tensile_test .solve_problem()
        print()


def test_nonlinear_shear_test():
    for bc_type in ("displacement", "traction"):
        shear_test = NonlinearShearTest(25, bc_type=bc_type)
        print(f"Running {shear_test._problem_name} with {bc_type} boundary condition type.")
        shear_test.solve_problem()
        print()

def test_nonlinear_body_force():
    body_force_test = NonlinearBodyForceTest(25)
    print(f"Running {body_force_test._problem_name}.")
    body_force_test.solve_problem()
    print()


def test_nonlinear_bc_function():
    bc_function_test = NonlinearBCFunctionTest(25)
    print(f"Running {bc_function_test._problem_name}.")
    bc_function_test.solve_problem()
    print()


if __name__ == "__main__":
    test_tensile_test()
    test_shear_test()
    test_body_force()
    test_bc_function()
    test_nonlinear_tensile_test()
    test_nonlinear_shear_test()
    test_nonlinear_body_force()
    test_nonlinear_bc_function()