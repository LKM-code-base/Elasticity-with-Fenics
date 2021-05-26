#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from os import path
from math import isfinite
import dolfin as dlfn
from auxiliary_methods import compute_elasticity_coefficients
from auxiliary_methods import ElasticModuli
from elastic_solver import LinearElasticitySolver, NonlinearElasticitySolver 


class ProblemBase:
    _suffix = ".xdmf"

    def __init__(self, elastic_law, main_dir=None):
        
        self._elastic_law = elastic_law
        
        # set write and read directory
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert path.exist(main_dir)
            self._main_dir = main_dir
        self._results_dir = path.join(self._main_dir, f"results/{self._elastic_law._linearity_type}/{self._elastic_law._name}")

    def _add_to_field_output(self, field):
        """
        Add the field to a list containing additional fields which are written
        to the xdmf file.
        """
        if not hasattr(self, "_additional_field_output"):
            self._additional_field_output = []
        self._additional_field_output.append(field)

    def _compute_stress_tensor(self):  # pragma: no cover
        """
        Returns the stress tensor.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _compute_pressure(self):  # pragma: no cover
        """
        Returns the pressure.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _compute_strain_tensor(self):  # pragma: no cover
        """
        Returns the strain tensor.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _get_boundary_conditions_map(self, field="displacement"):
        """
        Returns a mapping relating the type of the boundary condition to the
        boundary identifiers where it is is applied.
        """
        assert hasattr(self, "_bcs")

        bc_map = {}
        bcs = self._bcs[field]

        for bc_type, bc_bndry_id, _ in bcs:
            if bc_type in bc_map:
                tmp = list(bc_map[bc_type])
                tmp.append(bc_bndry_id)
                bc_map[bc_type] = tuple(tmp)
            else:
                bc_map[bc_type] = (bc_bndry_id, )

        return bc_map

    def _get_filename(self):  # pragma: no cover
        """
        Purely virtual method for setting the filename.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _get_solver(self):  # pragma: no cover
        """
        Purely virtual method for getting the solver of the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _write_xdmf_file(self, current_time=0.0):
        """
        Write the output to an xdmf file. The solution and additional fields
        are output to the file.
        """
        assert isinstance(current_time, float)

        # get filename
        fname = self._get_filename()
        assert fname.endswith(".xdmf")

        # create results directory
        assert hasattr(self, "_results_dir")
        if not path.exists(self._results_dir):
            os.makedirs(self._results_dir)

        # get solution
        solver = self._get_solver()
        solution = solver.solution

        # serialize
        with dlfn.XDMFFile(fname) as results_file:
            results_file.parameters["flush_output"] = True
            results_file.parameters["functions_share_mesh"] = True

            sub_space_association = solver.sub_space_association
            if len(sub_space_association) > 1:
                solution_components = solution.split()
                for index, name in solver.sub_space_association.items():
                    solution_components[index].rename(name, "")
                    results_file.write(solution_components[index], current_time)

            else:
                solution.rename(sub_space_association[0], "")
                results_file.write(solution, current_time)

            if hasattr(self, "_additional_field_output"):
                for field in self._additional_field_output:
                    results_file.write(field, current_time)

    def postprocess_solution(self):  # pragma: no cover
        """
        Virtual method for additional post-processing.
        """
        pass

    def setup_mesh(self):  # pragma: no cover
        """
        Purely virtual method for setting up the mesh of the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def set_boundary_conditions(self):  # pragma: no cover
        """
        Purely virtual method for specifying the boundary conditions of the
        problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def set_body_force(self):  # pragma: no cover
        """
        Virtual method for specifying the body force of the problem.
        """
        pass

    def solve_problem(self):  # pragma: no cover
        """
        Purely virtual method for solving the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    @property
    def space_dim(self):
        assert hasattr(self, "_space_dim")
        return self._space_dim


class LinearElasticProblem(ProblemBase):
    """
    Class to simulate a linear elastic problem using the
    `LinearElasticitySolver`.

    Parameters
    ----------
    main_dir: str (optional)
        Directory to save the results.
    tol: float (optional)
        Final tolerance.
    maxiter: int (optional)
        Maximum number of iterations in total.
    """
    def __init__(self, elastic_law, main_dir=None, tol=1e-10, maxiter=50):
        """
        Constructor of the class.
        """
        super().__init__(elastic_law, main_dir)

        # input check
        assert isinstance(maxiter, int) and maxiter > 0
        assert isinstance(tol, float) and tol > 0.0

        # set numerical tolerances
        self._tol = tol
        self._maxiter = maxiter

    def _compute_stress_tensor(self):
        """
        Returns the stress tensor.
        """
        assert hasattr(self, "_C")
        solver = self._get_solver()
        # displacement vector
        displacement = solver.solution
       
        stress = self._elastic_law.postprocess_cauchy_stress(displacement)
        # create function space
        family = displacement.ufl_element().family()
        assert family == "Lagrange"
        degree = displacement.ufl_element().degree()
        assert degree >= 0
        cell = self._mesh.ufl_cell()
        elemSigma = dlfn.TensorElement("DG", cell, degree - 1,
                                       shape=(self._space_dim, self._space_dim),
                                       symmetry=True)
        Wh = dlfn.FunctionSpace(self._mesh, elemSigma)

        # project
        sigma = dlfn.project(stress, Wh)
        sigma.rename("sigma", "")

        return sigma

    def _get_filename(self):
        """
        Class method returning a filename.
        """
        # input check
        assert hasattr(self, "_problem_name")
        problem_name = self._problem_name

        fname = problem_name
        fname += self._suffix

        return path.join(self._results_dir, fname)

    def _get_solver(self):
        assert hasattr(self, "_linear_elastic_solver")
        return self._linear_elastic_solver

    def set_parameters(self, **kwargs):
        """
        Sets up the parameters of the model by creating or modifying class
        objects.
        """
        if "C" in kwargs.keys():
            # 1st dimensionless coefficient
            C = kwargs["C"]
            assert isinstance(C, float)
            assert isfinite(C)
            assert C > 0.0
            self._C = C
            # 2nd dimensionless coefficient
            if "D" in kwargs.keys():
                D = kwargs["D"]
                assert isinstance(D, float)
                assert isfinite(D)
                assert D > 0.0
                self._D = D

        else:
            # extract elastic moduli
            cleaned_kwargs = kwargs.copy()
            if "lref" in cleaned_kwargs.keys():
                cleaned_kwargs.pop("lref")
            if "bref" in cleaned_kwargs.keys():
                cleaned_kwargs.pop("bref")
            elastic_moduli = compute_elasticity_coefficients(**cleaned_kwargs)
            lmbda = elastic_moduli[ElasticModuli.FirstLameParameter]
            mu = elastic_moduli[ElasticModuli.ShearModulus]
            # 1st dimensionless coefficient
            self._C = lmbda / mu

            # 2nd optional dimensionless coefficient
            if "lref" in kwargs.keys() and "bref" in kwargs.keys():
                # reference length
                lref = kwargs["lref"]
                assert isinstance(lref, float)
                assert isfinite(lref)
                assert lref > 0.0
                # reference value for the body force density
                bref = kwargs["bref"]
                assert isinstance(bref, float)
                assert isfinite(bref)
                assert bref > 0.0
                # 2nd optional dimensionless coefficient
                self._D = bref * lref / mu

            else:
                self._D = None

    def write_boundary_markers(self):
        """
        Write the boundary markers specified by the MeshFunction
        `_boundary_markers` to a pvd-file.
        """
        assert hasattr(self, "_boundary_markers")
        assert hasattr(self, "_problem_name")

        # create results directory
        assert hasattr(self, "_results_dir")
        if not path.exists(self._results_dir):
            os.makedirs(self._results_dir)

        problem_name = self._problem_name
        suffix = ".pvd"
        fname = problem_name + "_BoundaryMarkers"
        fname += suffix
        fname = path.join(self._results_dir, fname)

        dlfn.File(fname) << self._boundary_markers

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
        if not hasattr(self, "_linear_elastic_solver"):
            self._linear_elastic_solver = \
                LinearElasticitySolver(self._mesh, self._boundary_markers, self._elastic_law)

        # pass boundary conditions
        self._linear_elastic_solver.set_boundary_conditions(self._bcs)

        # pass dimensionless numbers
        if hasattr(self, "_D"):
            self._linear_elastic_solver.set_dimensionless_numbers(self._C,
                                                                  self._D)
        else:
            self._linear_elastic_solver.set_dimensionless_numbers(self._C)

        # pass body force
        if hasattr(self, "_body_force"):
            self._linear_elastic_solver.set_body_force(self._body_force)

        # solve problem
        if self._D is not None:
            dlfn.info("Solving problem with C = {0:.2f} and "
                      "D = {1:0.2f}".format(self._C, self._D))
        else:
            dlfn.info("Solving problem with C = {0:.2f}".format(self._C))
        self._linear_elastic_solver.solve()

        # postprocess solution
        self.postprocess_solution()

        # write XDMF-files
        self._write_xdmf_file()

class NonlinearElasticProblem(ProblemBase):
    """
    Class to simulate a nonlinear elastic problem using the
    `NonlinearElasticitySolver`.

    Parameters
    ----------
    main_dir: str (optional)
        Directory to save the results.
    tol: float (optional)
        Final tolerance.
    maxiter: int (optional)
        Maximum number of iterations in total.
    """
    def __init__(self, elastic_law, main_dir=None, tol=1e-10, maxiter=50):
        """
        Constructor of the class.
        """
        super().__init__(elastic_law, main_dir)

        # input check
        assert isinstance(maxiter, int) and maxiter > 0
        assert isinstance(tol, float) and tol > 0.0

        # set numerical tolerances
        self._tol = tol
        self._maxiter = maxiter

    def _compute_stress_tensor(self):
        """
        Returns the stress tensor.
        """
        assert hasattr(self, "_C")
        solver = self._get_solver()
        # displacement vector
        displacement = solver.solution
       
        stress = self._elastic_law.postprocess_cauchy_stress(displacement)
        # create function space
        family = displacement.ufl_element().family()
        assert family == "Lagrange"
        degree = displacement.ufl_element().degree()
        assert degree >= 0
        cell = self._mesh.ufl_cell()
        elemSigma = dlfn.TensorElement("DG", cell, degree - 1,
                                       shape=(self._space_dim, self._space_dim),
                                       symmetry=True)
        Wh = dlfn.FunctionSpace(self._mesh, elemSigma)

        # project
        sigma = dlfn.project(stress, Wh)
        sigma.rename("sigma", "")

        return sigma

    def _get_filename(self):
        """
        Class method returning a filename.
        """
        # input check
        assert hasattr(self, "_problem_name")
        problem_name = self._problem_name

        fname = problem_name
        fname += self._suffix

        return path.join(self._results_dir, fname)

    def _get_solver(self):
        assert hasattr(self, "_nonlinear_elastic_solver")
        return self._nonlinear_elastic_solver

    def set_parameters(self, **kwargs):
        """
        Sets up the parameters of the model by creating or modifying class
        objects.
        """
        if "C" in kwargs.keys():
            # 1st dimensionless coefficient
            C = kwargs["C"]
            assert isinstance(C, float)
            assert isfinite(C)
            assert C > 0.0
            self._C = C
            # 2nd dimensionless coefficient
            if "D" in kwargs.keys():
                D = kwargs["D"]
                assert isinstance(D, float)
                assert isfinite(D)
                assert D > 0.0
                self._D = D

        else:
            # extract elastic moduli
            cleaned_kwargs = kwargs.copy()
            if "lref" in cleaned_kwargs.keys():
                cleaned_kwargs.pop("lref")
            if "bref" in cleaned_kwargs.keys():
                cleaned_kwargs.pop("bref")
            elastic_moduli = compute_elasticity_coefficients(**cleaned_kwargs)
            lmbda = elastic_moduli[ElasticModuli.FirstLameParameter]
            mu = elastic_moduli[ElasticModuli.ShearModulus]
            # 1st dimensionless coefficient
            self._C = lmbda / mu

            # 2nd optional dimensionless coefficient
            if "lref" in kwargs.keys() and "bref" in kwargs.keys():
                # reference length
                lref = kwargs["lref"]
                assert isinstance(lref, float)
                assert isfinite(lref)
                assert lref > 0.0
                # reference value for the body force density
                bref = kwargs["bref"]
                assert isinstance(bref, float)
                assert isfinite(bref)
                assert bref > 0.0
                # 2nd optional dimensionless coefficient
                self._D = bref * lref / mu

            else:
                self._D = None

    def write_boundary_markers(self):
        """
        Write the boundary markers specified by the MeshFunction
        `_boundary_markers` to a pvd-file.
        """
        assert hasattr(self, "_boundary_markers")
        assert hasattr(self, "_problem_name")

        # create results directory
        assert hasattr(self, "_results_dir")
        if not path.exists(self._results_dir):
            os.makedirs(self._results_dir)

        problem_name = self._problem_name
        suffix = ".pvd"
        fname = problem_name + "_BoundaryMarkers"
        fname += suffix
        fname = path.join(self._results_dir, fname)

        dlfn.File(fname) << self._boundary_markers

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
        if not hasattr(self, "_nonlinear_elastic_solver"):
            self._nonlinear_elastic_solver = \
                NonlinearElasticitySolver(self._mesh, self._boundary_markers, self._elastic_law)

        # pass boundary conditions
        self._nonlinear_elastic_solver.set_boundary_conditions(self._bcs)

        # pass dimensionless numbers
        if hasattr(self, "_D"):
            self._nonlinear_elastic_solver.set_dimensionless_numbers(self._C,
                                                                  self._D)
        else:
            self._nonlinear_elastic_solver.set_dimensionless_numbers(self._C)

        # pass body force
        if hasattr(self, "_body_force"):
            self._nonlinear_elastic_solver.set_body_force(self._body_force)

        # solve problem
        if self._D is not None:
            dlfn.info("Solving problem with C = {0:.2f} and "
                      "D = {1:0.2f}".format(self._C, self._D))
        else:
            dlfn.info("Solving problem with C = {0:.2f}".format(self._C))
        self._nonlinear_elastic_solver.solve()

        # postprocess solution
        self.postprocess_solution()

        # write XDMF-files
        self._write_xdmf_file()