#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import PointSubDomain
from grid_generator import hyper_cube, cylinder
from grid_generator import HyperCubeBoundaryMarkers, CylinderBoundaryMarkers
from elastic_problem import CompressibleElasticProblem
from elastic_solver import DisplacementBCType
from elastic_solver import TractionBCType
from elastic_law import Hooke, StVenantKirchhoff, NeoHooke
import dolfin as dlfn
import numpy as np
import sympy as sp


class TensileTest(CompressibleElasticProblem):
    def __init__(self, n_points, elastic_law, main_dir=None, bc_type="floating"):
        super().__init__(elastic_law, main_dir)


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
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


class ShearTest(CompressibleElasticProblem):
    def __init__(self, n_points, elastic_law, main_dir=None, bc_type="displacement"):
        super().__init__(elastic_law, main_dir)

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
        self._bcs.append((DisplacementBCType.fixed, HyperCubeBoundaryMarkers.bottom.value, None)),

        if self._bc_type == "displacement":
            self._bcs.append((DisplacementBCType.constant, HyperCubeBoundaryMarkers.top.value, (0.1, 0.0)))

        elif self._bc_type == "traction":
            self._bcs.append((DisplacementBCType.fixed_component, HyperCubeBoundaryMarkers.top.value, 1, None))
            self._bcs.append((TractionBCType.constant_component, HyperCubeBoundaryMarkers.top.value, 0, 0.1))

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
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


class BodyForceTest(CompressibleElasticProblem):
    def __init__(self, n_points, elastic_law, main_dir=None):
        super().__init__(elastic_law, main_dir)

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
        self._bcs = [(DisplacementBCType.fixed, HyperCubeBoundaryMarkers.left.value, None),
                     (DisplacementBCType.fixed, HyperCubeBoundaryMarkers.right.value, None),
                     (DisplacementBCType.fixed, HyperCubeBoundaryMarkers.bottom.value, None),
                     (DisplacementBCType.fixed, HyperCubeBoundaryMarkers.top.value, None)]

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


class BCFunctionTest(CompressibleElasticProblem):
    def __init__(self, n_points, elastic_law, main_dir=None):
        super().__init__(elastic_law, main_dir)

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
        self._bcs = [(DisplacementBCType.fixed, HyperCubeBoundaryMarkers.left.value, None),
                     (DisplacementBCType.function, HyperCubeBoundaryMarkers.right.value, fun01),
                     (DisplacementBCType.function_component, HyperCubeBoundaryMarkers.top.value, 1, fun02)]

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


class CylinderTest(CompressibleElasticProblem):
    def __init__(self, n_points, elastic_law, top_displacement=0.1, dim=3, main_dir=None):
        super().__init__(elastic_law, main_dir)

        assert isinstance(dim, int)
        self._space_dim = dim

        self._n_points = n_points
        self._problem_name = "CylinderTest"

        self.set_parameters(E=210.0, nu=0.3)

        self._top_displacement = top_displacement

    def setup_mesh(self):
        # create mesh
        if self._space_dim == 2:
            self._mesh, self._boundary_markers = cylinder(self._space_dim, (0.1, 0.1), 1.0, 4)
        elif self._space_dim == 3:
            self._mesh, self._boundary_markers = cylinder(self._space_dim, (0.1, 0.1), 1.0, 1)

    def set_boundary_conditions(self):

        if self._space_dim == 2:
            # boundary conditions
            gamma01 = PointSubDomain((0.0, 0.0), tol=1e-10)
            gamma02 = dlfn.CompiledSubDomain("near(x[1], 0.0)")

            self._bcs = [(DisplacementBCType.fixed_pointwise, gamma01, None),
                         (DisplacementBCType.fixed_component_pointwise, gamma02, 1, None),
                         (DisplacementBCType.constant_component, CylinderBoundaryMarkers.top.value, 1, self._top_displacement)]

        if self._space_dim == 3:
            # boundary conditions
            gamma01 = PointSubDomain((0.0, 0.0, 0.0), tol=1e-10)
            gamma02 = dlfn.CompiledSubDomain("near(x[2], 0.0)")
            gamma03 = PointSubDomain((0.0, 0.1, 0.0), tol=1e-10)

            self._bcs = [(DisplacementBCType.fixed_pointwise, gamma01, None),
                         (DisplacementBCType.fixed_component_pointwise, gamma02, 2, None),
                         (DisplacementBCType.fixed_component_pointwise, gamma03, 0, None),
                         (DisplacementBCType.constant_component, CylinderBoundaryMarkers.top.value, 2, self._top_displacement)]

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


class DirichletTest(CompressibleElasticProblem):
    def __init__(self, n_points, elastic_law, main_dir=None):
        super().__init__(elastic_law, main_dir)

        self._n_points = n_points
        self._problem_name = "DirichletTest"

        self.set_parameters(C=1.5, D=1.)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_analytical_displacement(self):
        self._u0 = 0.1
        self._analytical_displacement = dlfn.Expression(
            ("u0 * ((1 + x[0]) * pow(x[1],2) * pow(1 - x[1],2) \
             * sin(2 * pi * x[0]) * cos(3 * pi * x[1]) \
                + (4 * x[0] * x[1]) * (1 - x[1]))",
                "u0 * (x[0] * (1 - x[0]) * sin (2 * pi * x[1]))"),
            u0=self._u0, degree=2
        )

    def set_analytical_displacement_sympy(self):
        if not hasattr(self, "_u0"):
            self.set_analytical_displacement()

        x, y = sp.symbols("x[0] x[1]")
        self._coords = (x, y)
        self._analytical_displacement_sympy = (self._u0 * ((1 + x) * pow(y, 2) * pow((1 - y), 2)
                                               * sp.sin(2 * sp.pi * x) * sp.cos(3 * sp.pi * y)
                                               + (4 * x * y) * (1 - y)),
                                               self._u0 * x * (1 - x) * sp.sin(2 * sp.pi * y))

    def set_body_force(self):
        self.set_analytical_displacement_sympy()

        def laplace_sympy(u):
            temp = []
            for i in range(len(u)):
                temp.append(sum(sp.diff(u[i], coord, 2) for coord in self._coords))
            return np.array(temp)

        def div_sympy(u):
            if np.size(u) == 2:  # u is a tensor of first order
                return sum(sp.diff(u[i], self._coords[i]) for i in range(len(u)))
            elif np.size(u) == self._space_dim**2:  # u is a tensor of second order
                return np.array(
                    tuple(
                        sum(sp.diff(u[i, j], self._coords[j]) for j in range(len(self._coords))) for i in range(len(u[0, :]))
                    )
                )

        def grad_sympy(u):
            if np.size(u) == 1:  # u is a scalar
                gu = tuple(
                    sp.diff(u, coord) for coord in self._coords
                )
            elif np.size(u) == self._space_dim:  # u is a tensor of first order
                assert isinstance(
                    u, (tuple, list, np.ndarray)
                )
                gu = tuple(
                    tuple(
                        sp.diff(u[i], coord) for coord in self._coords) for i in range(len(u))
                )
            elif np.size(u) == self._space_dim**2:  # u is a tensor of second order
                gu = tuple(
                    tuple(
                        tuple(
                            sp.diff(u[i, j], coord) for coord in self._coords) for j in range(len(u[0, :]))) for i in range(len(u[0, :]))
                )
            return np.array(gu)

        # compute body_force as a sympy expression:
        if self._elastic_law.name == "Hooke":
            body_force_sympy = -(self._C + 1) * grad_sympy(div_sympy(self._analytical_displacement_sympy)) - \
                laplace_sympy(self._analytical_displacement_sympy)

            self._body_force = dlfn.Expression(
                tuple(sp.printing.ccode(body_force_sympy[i]) for i in range(len(body_force_sympy))), degree=2
            )

        elif self._elastic_law.name == "StVenantKirchhoff":
            EYE = sp.eye(self._space_dim)
            u = self._analytical_displacement_sympy
            H = grad_sympy(u)
            body_force_sympy = - np.tensordot(
                EYE + grad_sympy(u),
                self._C * (grad_sympy(div_sympy(u)) + 1 / 2 * grad_sympy(np.trace(np.tensordot(H, H.T, axes=1))))
                + (div_sympy(H) + div_sympy(H.T) + div_sympy(np.tensordot(H.T, H, axes=1))), axes=1)\
                - np.tensordot(grad_sympy(H), self._C
                               * (div_sympy(u) + 1 / 2 * np.tensordot(H, H)) * EYE + (H + H.T + np.tensordot(H.T, H, axes=1)), axes=2)

            self._body_force = dlfn.Expression(
                tuple(sp.printing.ccode(body_force_sympy[i]) for i in range(len(body_force_sympy))),
                degree=2
            )

    def set_displacement_right(self):
        if not hasattr(self, "_u0"):
            self.set_analytical_displacement()

        self._displacement_right = dlfn.Expression(
            ("u0 * 4 * x[1] * (1 - x[1])", "0"), u0=self._u0,
            degree=2
        )

    def set_boundary_conditions(self):
        self.set_displacement_right()
        # boundary conditions
        self._bcs = [(DisplacementBCType.fixed, HyperCubeBoundaryMarkers.left.value, None),
                     (DisplacementBCType.function, HyperCubeBoundaryMarkers.right.value, self._displacement_right),
                     (DisplacementBCType.fixed, HyperCubeBoundaryMarkers.bottom.value, None),
                     (DisplacementBCType.fixed, HyperCubeBoundaryMarkers.top.value, None)]

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


def test_tensile_test():
    for elastic_law in [Hooke(), StVenantKirchhoff(), NeoHooke()]:
        for bc_type in ("floating", "clamped", "clamped_free", "pointwise"):
            tensile_test = TensileTest(25, elastic_law, bc_type=bc_type)
            print(f"Running {tensile_test._problem_name} with {bc_type} boundary condition type.")
            tensile_test.solve_problem()
            print()


def test_shear_test():
    for elastic_law in [Hooke(), StVenantKirchhoff(), NeoHooke()]:
        for bc_type in ("displacement", "traction"):
            shear_test = ShearTest(25, elastic_law, bc_type=bc_type)
            print(f"Running {shear_test._problem_name} with {bc_type} boundary condition type.")
            shear_test.solve_problem()
            print()



def test_body_force():
    for elastic_law in [Hooke(), StVenantKirchhoff(), NeoHooke()]:
        body_force_test = BodyForceTest(25, elastic_law)
        print(f"Running {body_force_test._problem_name}.")
        body_force_test.solve_problem()
        print()


def test_bc_function():
    for elastic_law in [Hooke(), StVenantKirchhoff()]:
        bc_function_test = BCFunctionTest(25, elastic_law)
        print(f"Running {bc_function_test._problem_name}.")
        bc_function_test.solve_problem()
        print()


def test_cylinder():
    cylinder_test = CylinderTest(25, StVenantKirchhoff(), top_displacement=-0.1, dim=2)
    print(f"Running {cylinder_test._problem_name} with top displacemt {cylinder_test._top_displacement}.")
    cylinder_test.solve_problem()
    print()


def test_cylinder_iteration(dim=2, elastic_law=StVenantKirchhoff()):
    displacements = np.linspace(-0.6, 2.0, num=10)
    stresses = []

    for displacement in displacements:

        cylinder_test = CylinderTest(25, elastic_law, top_displacement=displacement, dim=dim)
        print(f"Running {cylinder_test._problem_name} with top displacemt {cylinder_test._top_displacement}.")
        cylinder_test.solve_problem()
        print()

        stress_tensor = cylinder_test._compute_stress_tensor()
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=cylinder_test._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(cylinder_test.space_dim):
            for j in range(cylinder_test.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))

        avg_stress = dlfn.assemble(stress_tensor[cylinder_test._space_dim - 1, cylinder_test._space_dim - 1] * dV) / V
        stresses.append(avg_stress)

    print("Displacements:")
    print(displacements)
    print("Avg. stress:")
    print(stresses)


def test_dirichlet():
    for elastic_law in (Hooke(), StVenantKirchhoff()):
        errors = []
        dofs = []
        n_points = (10, 20, 30, 40)
        for n in n_points:
            dirichlet_test = DirichletTest(n, elastic_law)
            dirichlet_test.solve_problem()
            dirichlet_test.set_analytical_displacement()

            u_ana = dlfn.project(
                dirichlet_test._analytical_displacement, dirichlet_test._get_solver()._Vh
            )

            errors.append(
                dlfn.errornorm(
                    u_ana, dirichlet_test._get_solver().solution,
                    'L2'
                )
            )
            dofs.append(
                dirichlet_test._get_solver()._Vh.dim()
            )
        print(f'>>> Elastc law: {dirichlet_test._elastic_law.name}.')
        [print(f'>>> With {dofs[i]} DoFs the error is {errors[i]}') for i in range(len(n_points))]
        print()


if __name__ == "__main__":
    test_tensile_test()
    test_shear_test()
    test_body_force()
    test_bc_function()
    test_cylinder()
    test_dirichlet()
