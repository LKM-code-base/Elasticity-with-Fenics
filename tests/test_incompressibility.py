#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import PointSubDomain
from grid_generator import hyper_cube, hyper_rectangle, spherical_shell, half_spherical_shell, tire
from grid_generator import HyperCubeBoundaryMarkers, SphericalAnnulusBoundaryMarkers, SphericalHalfAnnulusBoundaryMarkers
from elastic_problem import ElasticProblem
from elastic_solver import DisplacementBCType, TractionBCType, ElasticitySolver
from elastic_law import NeoHooke, NeoHookeIncompressible, MooneyRivlinIncompressible
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
        # compute volume ratio
        J = self._compute_volume_ratio()
        # add stress components to the field output
        component_indices = []
        for i in range(self.space_dim):
            for j in range(i, self.space_dim):
                component_indices.append((i + 1, j + 1))
        for k, stress in enumerate(stress_tensor.split()):
            stress.rename("S{0}{1}".format(*component_indices[k]), "")
            self._add_to_field_output(stress)
        self._add_to_field_output(J)
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


class HyperRectangleTest(ElasticProblem):
    # def init(self, elastic_law, n_points=25, top_displacement=0.1, dim=3, main_dir=None, polynomial_degree=2):
    #    super().init(elastic_law, main_dir, polynomial_degree=polynomial_degree)
    def __init__(self, n_points, elastic_law, main_dir=None, dim=3, top_displacement=0.1, polynomial_degree=2):
        super().__init__(elastic_law, main_dir, polynomial_degree=polynomial_degree)

        assert isinstance(dim, int)
        self._space_dim = dim

        self._n_points = n_points
        self._problem_name = "HyperRectangleTest"

        self.set_parameters(E=210.0, nu=0.3)

        self._top_displacement = top_displacement

    def setup_mesh(self):
        # create mesh
        if self._space_dim == 2:
            self._mesh, self._boundary_markers = hyper_rectangle((-0.05, 0.0), (0.05, 1.0))
        elif self._space_dim == 3:
            self._mesh, self._boundary_markers = hyper_rectangle((-0.05, -0.05, 0.0), (0.05, 0.05, 1.0))

    def set_boundary_conditions(self):

        if self._space_dim == 2:
            # boundary conditions
            gamma01 = PointSubDomain((0.0, 0.0), tol=1e-10)
            gamma02 = dlfn.CompiledSubDomain("near(x[1], 0.0)")

            self._bcs = [(DisplacementBCType.fixed_pointwise, gamma01, None),
                         (DisplacementBCType.fixed_component_pointwise, gamma02, 1, None),
                         (DisplacementBCType.constant_component, HyperCubeBoundaryMarkers.top.value, 1, self._top_displacement)]

        if self._space_dim == 3:
            # boundary conditions
            gamma01 = PointSubDomain((0.0, 0.0, 0.0), tol=1e-10)
            gamma02 = dlfn.CompiledSubDomain("near(x[2], 0.0)")
            gamma03 = PointSubDomain((0.0, 0.1, 0.0), tol=1e-10)

            self._bcs = [(DisplacementBCType.fixed_pointwise, gamma01, None),
                         (DisplacementBCType.fixed_component_pointwise, gamma02, 2, None),
                         (DisplacementBCType.fixed_component_pointwise, gamma03, 0, None),
                         (DisplacementBCType.constant_component, HyperCubeBoundaryMarkers.front.value, 2, self._top_displacement)]

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


class BalloonTest(ElasticProblem):
    def __init__(self, n_refinments, elastic_law, main_dir=None, dim=3, polynomial_degree=2):
        super().__init__(elastic_law, main_dir, polynomial_degree=polynomial_degree)

        assert isinstance(dim, int)
        self._space_dim = dim

        self._n_refinements = n_refinments
        self._problem_name = "BalloonTest"

        self.set_parameters(E=1.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        if self._space_dim == 2:
            self._mesh, self._boundary_markers = spherical_shell(2, (0.9, 1.0), self._n_refinements)
        elif self._space_dim == 3:
            self._mesh, self._boundary_markers = spherical_shell(3, (0.9, 1.0), self._n_refinements)

    def set_boundary_conditions(self):
        if self._space_dim == 2:
            gamma01 = PointSubDomain((0.0, -1.0), tol=1e-10)
            self._bcs = [(DisplacementBCType.fixed_pointwise, gamma01, None),
                         (TractionBCType.constant_pressure, SphericalAnnulusBoundaryMarkers.interior_boundary.value, - 0.2),
                         (TractionBCType.constant_pressure, SphericalAnnulusBoundaryMarkers.exterior_boundary.value, - 0.1)]

        if self._space_dim == 3:
            gamma01 = PointSubDomain((0.0, 0.0, -1.0), tol=1e-10)
            self._bcs = [(DisplacementBCType.fixed_pointwise, gamma01, None),
                         (TractionBCType.constant_pressure, SphericalAnnulusBoundaryMarkers.interior_boundary.value, - 0.2),
                         (TractionBCType.constant_pressure, SphericalAnnulusBoundaryMarkers.exterior_boundary.value, - 0.1)]

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
        self._add_to_field_output(self._compute_pressure())
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


class HalfBalloonTest(ElasticProblem):
    def __init__(self, n_refinments, elastic_law, main_dir=None, dim=3, polynomial_degree=2):
        super().__init__(elastic_law, main_dir, polynomial_degree=polynomial_degree)

        assert isinstance(dim, int)
        self._space_dim = dim

        self._n_refinements = n_refinments
        self._problem_name = "BalloonTest"

        self.set_parameters(E=1.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        if self._space_dim == 2:
            self._mesh, self._boundary_markers = half_spherical_shell(2, (0.5, 1.0), self._n_refinements)
        elif self._space_dim == 3:
            self._mesh, self._boundary_markers = half_spherical_shell(3, (0.9, 1.0), self._n_refinements)

    def set_boundary_conditions(self):
        if self._space_dim == 2:
            gamma01 = PointSubDomain((0.0, 1.0), tol=1e-10)
            self._bcs = [(DisplacementBCType.fixed_component_pointwise, gamma01, 0, None),
                         (DisplacementBCType.fixed_component, SphericalHalfAnnulusBoundaryMarkers.bottom_boundary.value, 1, None),
                         (TractionBCType.constant_pressure, SphericalHalfAnnulusBoundaryMarkers.interior_boundary.value, -0.1)]

        if self._space_dim == 3:
            gamma01 = PointSubDomain((1.0, 0.0, 0.0), tol=1e-10)
            gamma03 = PointSubDomain((0.0, 1.0, 0.0), tol=1e-10)
            self._bcs = [(DisplacementBCType.fixed_component_pointwise, gamma01, 1, None),
                         (DisplacementBCType.fixed_component_pointwise, gamma03, 0, None),
                         (DisplacementBCType.fixed_component, SphericalHalfAnnulusBoundaryMarkers.bottom_boundary.value, 2, None),
                         (TractionBCType.constant_pressure, SphericalHalfAnnulusBoundaryMarkers.interior_boundary.value, -0.1)]

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
        self._add_to_field_output(self._compute_pressure())
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))
        """
        # compute surface average of the Lagrange Multiplier
        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)
        A_inner = dlfn.assemble(dlfn.Constant(1.0) * dA(SphericalHalfAnnulusBoundaryMarkers.interior_boundary.value))
        A_outer = dlfn.assemble(dlfn.Constant(1.0) * dA(SphericalHalfAnnulusBoundaryMarkers.exterior_boundary.value))
        solver = self._get_solver()
        displacement, pressure = solver.solution.split(True)
        print("Volume-averaged Lagrange Multiplier: ")
        avg_pressure_inner = dlfn.assemble(pressure * dA(SphericalHalfAnnulusBoundaryMarkers.interior_boundary.value)) / A_inner
        avg_pressure_outer = dlfn.assemble(pressure * dA(SphericalHalfAnnulusBoundaryMarkers.exterior_boundary.value)) / A_outer
        print("Avg. Lagrange Multiplier inner:")
        print(avg_pressure_inner)
        print("Avg. Lagrange Multiplier outer:")
        print(avg_pressure_outer)
        print()
        """


class IterativeScalingHalfBalloonTest(ElasticProblem):
    def __init__(self, n_refinments, elastic_law, main_dir=None, dim=3, polynomial_degree=2):
        super().__init__(elastic_law, main_dir, polynomial_degree=polynomial_degree)

        assert isinstance(dim, int)
        self._space_dim = dim

        self._n_refinements = n_refinments
        self._problem_name = "IterativeScalingBalloonTest"

        self.set_parameters(E=1.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        if self._space_dim == 2:
            self._mesh, self._boundary_markers = half_spherical_shell(2, (0.5, 1.0), self._n_refinements)
        elif self._space_dim == 3:
            self._mesh, self._boundary_markers = half_spherical_shell(3, (0.9, 1.0), self._n_refinements)

    def set_boundary_conditions(self):
        if self._space_dim == 2:
            gamma01 = PointSubDomain((0.0, 1.0), tol=1e-10)
            self._bcs = [(DisplacementBCType.fixed_component_pointwise, gamma01, 0, None),
                         (DisplacementBCType.fixed_component, SphericalHalfAnnulusBoundaryMarkers.bottom_boundary.value, 1, None),
                         (TractionBCType.constant_pressure, SphericalHalfAnnulusBoundaryMarkers.exterior_boundary.value, 0.0),
                         (TractionBCType.function_pressure, SphericalHalfAnnulusBoundaryMarkers.interior_boundary.value, dlfn.Expression("-scaling_factor * .6", scaling_factor=1.1, degree=0))]

        if self._space_dim == 3:
            gamma01 = PointSubDomain((1.0, 0.0, 0.0), tol=1e-10)
            gamma03 = PointSubDomain((0.0, 1.0, 0.0), tol=1e-10)
            self._bcs = [(DisplacementBCType.fixed_component_pointwise, gamma01, 1, None),
                         (DisplacementBCType.fixed_component_pointwise, gamma03, 0, None),
                         (DisplacementBCType.fixed_component, SphericalHalfAnnulusBoundaryMarkers.bottom_boundary.value, 2, None),
                         (TractionBCType.function_pressure, SphericalHalfAnnulusBoundaryMarkers.interior_boundary.value, dlfn.Expression(("-scaling_factor * 0.1"), scaling_factor=0.0, degree=0))]

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
        self._add_to_field_output(self._compute_pressure())
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))

        # compute surface average of the Lagrange Multiplier
        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)
        A_inner = dlfn.assemble(dlfn.Constant(1.0) * dA(SphericalHalfAnnulusBoundaryMarkers.interior_boundary.value))
        A_outer = dlfn.assemble(dlfn.Constant(1.0) * dA(SphericalHalfAnnulusBoundaryMarkers.exterior_boundary.value))
        solver = self._get_solver()
        displacement, pressure = solver.solution.split(True)
        print("Volume-averaged Lagrange Multiplier: ")
        avg_pressure_inner = dlfn.assemble(pressure * dA(SphericalHalfAnnulusBoundaryMarkers.interior_boundary.value)) / A_inner
        avg_pressure_outer = dlfn.assemble(pressure * dA(SphericalHalfAnnulusBoundaryMarkers.exterior_boundary.value)) / A_outer
        print("Avg. Lagrange Multiplier inner:")
        print(avg_pressure_inner)
        print("Avg. Lagrange Multiplier outer:")
        print(avg_pressure_outer)
        print()

    def solve_problem(self):
        """
        Solve the stationary problem.
        """
        # setup mesh
        self.setup_mesh()
        assert self._mesh is not None
        self._space_dim = self._mesh.geometry().dim()
        self._n_cells = self._mesh.num_cells()

        # setup boundary conditions
        self.set_boundary_conditions()

        # setup body force
        self.set_body_force()

        # setup parameters
        if not hasattr(self, "_C"):
            self.set_parameters()

        # create solver object
        if not hasattr(self, "_elastic_solver"):
            self._elastic_solver = \
                ElasticitySolver(self._mesh, self._boundary_markers, self._elastic_law, polynomial_degree=self._polynomial_degree)

        # pass boundary conditions
        self._elastic_solver.set_boundary_conditions(self._bcs)

        # pass dimensionless numbers
        if hasattr(self, "_D"):
            self._elastic_solver.set_dimensionless_numbers(self._C, self._D)
        else:
            self._elastic_solver.set_dimensionless_numbers(self._C)

        # pass body force
        if hasattr(self, "_body_force"):
            self._elastic_solver.set_body_force(self._body_force)

        # solve problem
        if self._D is not None:
            dlfn.info("Solving problem with C = {0:.2f} and "
                      "D = {1:0.2f}".format(self._C, self._D))
        else:
            dlfn.info("Solving problem with C = {0:.2f}".format(self._C))

        from os import getcwd
        ufile = dlfn.File(f"{getcwd()}/results/velocity.pvd")
        pfile = dlfn.File(f"{getcwd()}/results/pressure.pvd")
        import numpy as np
        linRange = np.linspace(0.0, 1.0, num=11, endpoint=True)
        print(linRange)
        for load_scaling in linRange:
            print(load_scaling)
            self._elastic_solver.set_scaling_factor(load_scaling)
            self._elastic_solver.solve()

            solver = self._get_solver()
            u, p = solver.solution.split(True)
            ufile << u
            pfile << p
        # postprocess solution
        self.postprocess_solution()

        # write XDMF-files
        self._write_xdmf_file()


class TireTest(ElasticProblem):
    def __init__(self, n_refinments, elastic_law, main_dir=None, dim=2, polynomial_degree=2):
        super().__init__(elastic_law, main_dir, polynomial_degree=polynomial_degree)

        assert isinstance(dim, int)
        self._space_dim = dim

        self._n_refinements = n_refinments
        self._problem_name = "TireTest"

        self.set_parameters(E=1.0, nu=0.3)

    def setup_mesh(self):
        # create mesh
        if self._space_dim == 2:
            self._mesh, self._boundary_markers = tire(2, self._n_refinements)
        if self._space_dim == 3:
            self._mesh, self._boundary_markers = tire(3, self._n_refinements)

    def set_boundary_conditions(self):

        if self._space_dim == 2:
            self._bcs = [(DisplacementBCType.fixed, 301, None),
                         (TractionBCType.constant_pressure, 100, - 0.01)]
        if self._space_dim == 3:
            self._bcs = [(DisplacementBCType.fixed, 301, None),
                         (DisplacementBCType.fixed_component, 1000, 2, None),
                         (DisplacementBCType.fixed_component, 2000, 1, None),
                         (TractionBCType.constant_pressure, 100, - 0.01)]

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
        self._add_to_field_output(self._compute_pressure())
        # compute volume average of the stress tensor
        dV = dlfn.Measure("dx", domain=self._mesh)
        V = dlfn.assemble(dlfn.Constant(1.0) * dV)
        print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i, j] * dV) / V
                print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))


def test_tensile_test():
    for elastic_law in [NeoHooke(), NeoHookeIncompressible(), MooneyRivlinIncompressible()]:
        for bc_type in ("floating", "clamped", "clamped_free", "pointwise"):
            tensile_test = TensileTest(25, elastic_law, bc_type=bc_type)
            print(f"Running {tensile_test._problem_name} with {bc_type} boundary condition type.")
            tensile_test.solve_problem()
            print()


def test_hyper_rectangle(top_displacement=0.1, dim=3):
    hyper_rectangle_test = HyperRectangleTest(25, NeoHookeIncompressible())
    print(f"Elastic law: {hyper_rectangle_test._elastic_law.name}")
    print(f"Running {hyper_rectangle_test._problem_name} with top displacemt {hyper_rectangle_test._top_displacement}.")
    hyper_rectangle_test.solve_problem()
    print()


def test_J_convergence():
    for elastic_law in [NeoHookeIncompressible()]:
        number_points = [10, 20, 30, 40, 50]
        errors_J = []
        mesh_sizes = []

        for points in number_points:
            tensile_test = TensileTest(points, elastic_law, bc_type="clamped")
            tensile_test.solve_problem()
            J = tensile_test._compute_volume_ratio()
            dV = dlfn.Measure("dx", domain=tensile_test._mesh)
            error_J = dlfn.assemble((J - dlfn.Constant(1.0)) ** 2 * dV)
            errors_J.append(error_J)
            mesh_sizes.append(tensile_test._mesh.hmax())

        print(mesh_sizes)
        print(errors_J)


def test_ballon(dim=2):
    ballon_test = BalloonTest(0, NeoHookeIncompressible(), dim=dim)
    ballon_test.solve_problem()


def test_half_ballon(dim=2):
    ballon_test = HalfBalloonTest(0, NeoHookeIncompressible(), dim=dim)
    ballon_test.solve_problem()


def test_scaling_half_ballon(dim=2):
    ballon_test = IterativeScalingHalfBalloonTest(0, NeoHookeIncompressible(), dim=dim)
    ballon_test.solve_problem()


def test_tire(dim=2):
    tire_test = TireTest(0, NeoHookeIncompressible(), dim=dim)
    tire_test.solve_problem()


if __name__ == "__main__":
    # test_tensile_test()
    # test_hyper_rectangle()
    # test_J_convergence()
    # test_half_ballon(dim=2)
    # test_scaling_half_ballon(dim=2)
    test_tire(dim=2)
