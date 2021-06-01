#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube
from grid_generator import hyper_cuboid
from grid_generator import HyperCubeBoundaryMarkers as BoundaryMarkers
from elastic_problem import LinearElasticProblem
from elastic_solver import DisplacementBCType
from elastic_solver import TractionBCType
import dolfin as dlfn


# Anfang ############################################################################
class WaveTest(LinearElasticProblem):
    # Initialisierung, hier werden folgendes deklariert:
    # - n_points: number of points (Knotenanzahl in einer Richtung, fängt jedoch vom Knoten 0 an) --> Für ein Gebiet aus einem Quader ist es ausreichend
    # - main_dir: Pfad
    # - bc_type: Art von Randbedingungen, default-Einstellung ist "floating"
    def __init__(self, n_points, main_dir=None, bc_type="floating"):
        super().__init__(main_dir)  # super() heißt, dass die Funktion __init__ die Variable main_dir von den höheren Klassenstufe vererbt bekommt (nicht so wichtig)
        
        # Abfrage ob die Bedingungen für Knotenanzahl zutreffend sind
        assert isinstance(n_points, int)
        assert n_points > 0
        self._n_points = n_points
        
        # Abfrage ob die Bedingungen für Art der Randbedingungen zutreffend sind
        assert isinstance(bc_type, str)
        # zunächst 1 R.B.:
        # - clamped: Die linken und rechten Seiten des Klotzs werden fest eigespannt, die unteren und oberen Seiten werden in y-Richtung losgelagert 
        assert bc_type in ("clamped")
        self._bc_type = bc_type

        if self._bc_type == "clamped":
            self._problem_name = "WaveTestClamped"
        
        # Geometrie des Zylinders
        L = 2.0
        D = 0.5
        self.r = D/L
        
        # Parameter-Eingabe
        #self.set_parameters(E=210.0e9, nu=0.3, rho=7800.0, lref=L, tend=1.0, numsteps=10) # tref = lref/ct
        self.set_parameters(E=1.0, nu=0.3, rho=0.5, lref=L, tend=10.0, numsteps=50) # tref = lref/ct
        
    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cuboid(2, self.r, self._n_points)
    
    def set_boundary_conditions(self):
        # boundary conditions
        self._bcs = []
        if self._bc_type == "clamped":
            self._bcs.append((DisplacementBCType.fixed, BoundaryMarkers.left.value, None))
            self._bcs.append((DisplacementBCType.fixed, BoundaryMarkers.right.value, None))
            self._bcs.append((DisplacementBCType.fixed_component, BoundaryMarkers.top.value, 1, None))
            self._bcs.append((DisplacementBCType.fixed_component, BoundaryMarkers.bottom.value, 1, None))

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
        #print("Volume-averaged stresses: ")
        for i in range(self.space_dim):
            for j in range(self.space_dim):
                avg_stress = dlfn.assemble(stress_tensor[i,j] * dV) / V
                #print("({0},{1}) : {2:8.2e}".format(i, j, avg_stress))
# Ende ############################################################################


class TensileTest(LinearElasticProblem):
    # Initialisierung, hier werden folgendes deklariert:
    # - n_points: number of points (Knotenanzahl in einer Richtung, fängt jedoch vom Knoten 0 an) --> Für ein Gebiet aus einem Quader ist es ausreichend
    # - main_dir: Pfad
    # - bc_type: Art von Randbedingungen, default-Einstellung ist "floating"
    def __init__(self, n_points, main_dir=None, bc_type="floating"):
        super().__init__(main_dir)  # super() heißt, dass die Funktion __init__ die Variable main_dir von den höheren Klassenstufe vererbt bekommt (nicht so wichtig)
        
        # Abfrage ob die Bedingungen für Knotenanzahl zutreffend sind
        assert isinstance(n_points, int)
        assert n_points > 0
        self._n_points = n_points
        
        # Abfrage ob die Bedingungen für Art der Randbedingungen zutreffend sind
        assert isinstance(bc_type, str)
        # zunächst 4 R.B.en: 
        # - floating: linke Seite des Klotzs ist in x-Richtung fest, untere Seite des Klotzs ist in y-Richtung fest, rechte Seite wird in x-Richtung gezogen
        # - clamped: linke Seite des Klotzs ist sowohl in x- als auch in y-Richtung fest (feste Einspannung), die rechte Seite wird mit der Verschiebung (0.1, 0.0) gezogen () 
        # - clamped_free: 
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
        if self._bc_type == "floating":
            self._bcs.append((DisplacementBCType.fixed_component, BoundaryMarkers.left.value, 0, None))
            self._bcs.append((DisplacementBCType.fixed_component, BoundaryMarkers.bottom.value, 1, None))
            self._bcs.append((DisplacementBCType.constant_component, BoundaryMarkers.right.value, 0, 0.1))
        elif self._bc_type == "clamped":
            self._bcs.append((DisplacementBCType.fixed, BoundaryMarkers.left.value, None))
            self._bcs.append((DisplacementBCType.constant, BoundaryMarkers.right.value, (0.1, 0.0)))
        elif self._bc_type == "clamped_free":
            self._bcs.append((DisplacementBCType.fixed, BoundaryMarkers.left.value, None))
            self._bcs.append((DisplacementBCType.constant_component, BoundaryMarkers.right.value, 0, 0.1))
        elif self._bc_type == "pointwise":
            raise NotImplementedError()

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


def test_wave_test():
    wave_test = WaveTest(n_points=25, bc_type="clamped")
    wave_test.solve_wave_problem()

def test_tensile_test():
    for bc_type in ("floating", "clamped", "clamped_free"):
        tensile_test = TensileTest(n_points=25, bc_type=bc_type)
        tensile_test.solve_problem()


def test_shear_test():
    for bc_type in ("displacement", "traction"):
        shear_test = ShearTest(n_points=25, bc_type=bc_type)
        shear_test.solve_problem()


def test_body_force():
    body_force_test = BodyForceTest(n_points=25)
    body_force_test.solve_problem()


def test_bc_function():
    bc_function_test = BCFunctionTest(n_points=25)
    bc_function_test.solve_problem()


if __name__ == "__main__":
    test_wave_test()