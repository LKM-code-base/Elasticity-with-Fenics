#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import isfinite
from enum import Enum, auto

import dolfin as dlfn
from dolfin import dot

from auxiliary_methods import extract_all_boundary_markers
from elastic_law import ElasticLaw


class DisplacementBCType(Enum):
    fixed = auto()
    fixed_component = auto()
    fixed_pointwise = auto()
    fixed_component_pointwise = auto()
    constant = auto()
    constant_component = auto()
    function = auto()
    function_component = auto()


class TractionBCType(Enum):
    constant = auto()
    constant_component = auto()
    function = auto()
    function_component = auto()
    free = auto()
    constant_pressure = auto()
    function_pressure = auto()


class SolverBase:
    """
    Base class for solvers.
    """
    _scaling_factor = 1.0
    _use_scaling_factor = False

    def __init__(self, mesh, boundary_markers, elastic_law, polynomial_degree=1):
        # input check
        assert isinstance(mesh, dlfn.Mesh)
        assert isinstance(boundary_markers, (dlfn.cpp.mesh.MeshFunctionSizet,
                                             dlfn.cpp.mesh.MeshFunctionInt))
        assert isinstance(polynomial_degree, int)
        assert polynomial_degree > 0
        # set mesh variables
        self._mesh = mesh
        self._boundary_markers = boundary_markers
        self._space_dim = self._mesh.geometry().dim()
        assert self._boundary_markers.dim() == self._space_dim - 1
        self._n_cells = self._mesh.num_cells()

        # dimension-dependent variables
        self._null_vector = dlfn.Constant((0., ) * self._space_dim)

        # set elastic law
        assert isinstance(elastic_law, ElasticLaw)
        self._elastic_law = elastic_law

        # set discretization parameters
        # polynomial degree
        self._p_deg = polynomial_degree
        # quadrature degree
        q_deg = 2 * self._p_deg
        dlfn.parameters["form_compiler"]["quadrature_degree"] = q_deg

    def _check_boundary_condition_format(self):
        """
        Virtual method to check the general format of an arbitrary boundary condition.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def set_body_force(self, body_force):
        """
        Specifies the body force.

        Parameters
        ----------
        body_force : dolfin.Expression, dolfin. Constant
            The body force.
        """
        assert isinstance(body_force, (dlfn.Expression, dlfn.Constant))
        # check rank of expression
        assert body_force.value_rank() == 1
        self._body_force = body_force

    def set_boundary_conditions(self):
        """
        Purely virtual method to set the boundary conditions of the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _setup_boundary_conditions(self):
        """
        Purely virtual method to set up the boundary conditions of the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def set_dimensionless_numbers(self):
        """
        Purely virtual method to update the parameters of the model by creating or modifying class
        objects.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _setup_function_spaces(self):
        """
        Virtual class method setting up function spaces.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _setup_problem(self):
        """
        Virtual method to set up solver objects.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def solve(self):
        """
        Purely virtual method for solving the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def set_scaling_factor(self, scaling_factor):
        assert isinstance(scaling_factor, float)
        self._use_scaling_factor = True
        self._scaling_factor = scaling_factor

    @property
    def sub_space_association(self):
        return self._sub_space_association

    @property
    def field_association(self):
        return self._field_association

    @property
    def solution(self):
        return self._solution


class ElasticitySolver(SolverBase):
    """
    Base class for compressible and incompressible elasticity.
    """
    # class variable
    _null_scalar = dlfn.Constant(0.)

    def __init__(self, mesh, boundary_markers, elastic_law, polynomial_degree=1):
        assert (elastic_law.compressiblity_type != "Incompressible" or polynomial_degree >= 2)

        if elastic_law.compressiblity_type == "Compressible":
            self._sub_space_association = {0: "displacement"}
        elif elastic_law.compressiblity_type == "Incompressible":
            self._sub_space_association = {0: "displacement", 1: "pressure"}

        self._field_association = {value: key for key, value in self._sub_space_association.items()}

        super().__init__(mesh, boundary_markers, elastic_law, polynomial_degree)

    def _check_boundary_condition_format(self, bc):
        """
        Check the general format of an arbitrary boundary condition.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")
        # boundary ids specified in the MeshFunction
        all_bndry_ids = extract_all_boundary_markers(self._mesh, self._boundary_markers)
        # 0. input check
        assert isinstance(bc, (list, tuple))
        assert len(bc) >= 2
        # 1. check bc type
        assert isinstance(bc[0], (DisplacementBCType, TractionBCType))
        # To Do: Changed it from rank = 1
        rank = 0
        # 2. check boundary id
        if bc[0] not in (DisplacementBCType.fixed_component_pointwise, DisplacementBCType.fixed_pointwise):
            assert isinstance(bc[1], int)
            assert bc[1] in all_bndry_ids, "Boundary id {0} ".format(bc[1]) +\
                                           "was not found in the boundary markers."
        else:
            assert isinstance(bc[1], dlfn.SubDomain)
        # 3. check value type
        # distinguish between scalar and vector field
        if rank == 0:
            # scalar field (tensor of rank zero)
            print(type(bc[2]))
            assert isinstance(bc[2], (dlfn.Expression, float)) or bc[2] is None
            if isinstance(bc[2], dlfn.Expression):
                # check rank of expression
                assert bc[2].value_rank() == 0

        elif rank == 1:
            # vector field (tensor of rank one)
            # distinguish between full or component-wise boundary conditions
            if len(bc) == 3:
                # full boundary condition
                assert isinstance(bc[2], (dlfn.Expression, tuple, list, float)) or bc[2] is None
                if isinstance(bc[2], dlfn.Expression):
                    # check rank of expression
                    assert bc[2].value_rank() == 1
                elif isinstance(bc[2], (tuple, list)):
                    # size of tuple or list
                    assert len(bc[2]) == self._space_dim
                    # type of the entries
                    assert all(isinstance(x, float) for x in bc[2])

            elif len(bc) == 4:
                # component-wise boundary condition
                # component index specified
                assert isinstance(bc[2], int)
                assert bc[2] < self._space_dim
                # value specified
                assert isinstance(bc[3], (dlfn.Expression, float)) or bc[3] is None
                if isinstance(bc[3], dlfn.Expression):
                    # check rank of expression
                    assert bc[3].value_rank() == 0

    def set_boundary_conditions(self, bcs):
        """
        Set the boundary conditions of the problem.
        The boundary conditions are specified as a list of tuples where each
        tuple represents a separate boundary condition. This means that, for
        example,
            bcs = [(Type, boundary_id, value),
                   (Type, boundary_id, component, value)]
        The first entry of each tuple specifies the type of the boundary
        condition. The second entry specifies the boundary identifier where the
        boundary should be applied. If full vector field is constrained through
        the boundary condition, the third entry specifies the value. If only a
        single component is constrained, the third entry specifies the
        component index and the third entry specifies the value.
        """
        assert isinstance(bcs, (list, tuple))
        # check format
        """
        # To Do: We don't know why this is not working.
        for bc in bcs:
            self._check_boundary_condition_format(bc)
        """
        # extract displacement/traction bcs and related boundary ids
        displacement_bcs = []
        displacement_bc_ids = set()
        traction_bcs = []
        traction_bc_ids = set()
        for bc in bcs:
            if isinstance(bc[0], DisplacementBCType):
                displacement_bcs.append(bc)
                displacement_bc_ids.add(bc[1])
            elif isinstance(bc[0], TractionBCType):
                traction_bcs.append(bc)
                traction_bc_ids.add(bc[1])
        # check that at least one displacement bc is specified
        assert len(displacement_bcs) > 0

        # check that there is no conflict between displacement and traction bcs
        if len(traction_bcs) > 0:
            # compute boundary ids with simultaneous bcs
            joint_bndry_ids = displacement_bc_ids.intersection(traction_bc_ids)
            # make sure that bcs are only applied component-wise
            allowedDisplacementBCTypes = (DisplacementBCType.fixed_component,
                                          DisplacementBCType.fixed_component_pointwise,
                                          DisplacementBCType.constant_component,
                                          DisplacementBCType.function_component)
            allowedTractionBCTypes = (TractionBCType.constant_component,
                                      TractionBCType.function_component,
                                      TractionBCType.constant_pressure,
                                      TractionBCType.function_pressure)

            for bndry_id in joint_bndry_ids:
                # extract component of displacement bc
                disp_bc_component = None
                for bc in displacement_bcs:
                    if bc[1] == bndry_id:
                        assert bc[0] in allowedDisplacementBCTypes
                        disp_bc_component = bc[2]
                        break
                # extract component of traction bc
                traction_bc_component = None
                for bc in traction_bcs:
                    if bc[1] == bndry_id:
                        assert bc[0] in allowedTractionBCTypes
                        traction_bc_component = bc[2]
                        break
                # compare components
                assert traction_bc_component != disp_bc_component
        # boundary conditions accepted
        self._displacement_bcs = displacement_bcs
        if len(traction_bcs) > 0:
            self._traction_bcs = traction_bcs

    def _setup_boundary_conditions(self):
        assert hasattr(self, "_Vh")
        assert hasattr(self, "_boundary_markers")
        assert hasattr(self, "_displacement_bcs")
        # empty dirichlet bcs
        self._dirichlet_bcs = []

        # displacement part
        for bc in self._displacement_bcs:
            # unpack values
            if len(bc) == 3:
                bc_type, bndry_id, value = bc
            elif len(bc) == 4:
                bc_type, bndry_id, component_index, value = bc
            else:  # pragma: no cover
                raise RuntimeError()
            # create dolfin.DirichletBC object
            if bc_type is DisplacementBCType.fixed:
                bc_object = dlfn.DirichletBC(self._Vh, self._null_vector,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.fixed_component:
                bc_object = dlfn.DirichletBC(self._Vh.sub(component_index),
                                             self._null_scalar,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.fixed_pointwise:
                assert isinstance(bndry_id, dlfn.SubDomain)
                bc_object = dlfn.DirichletBC(self._Vh, self._null_vector,
                                             bndry_id, "pointwise")
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.fixed_component_pointwise:
                assert isinstance(bndry_id, dlfn.SubDomain)
                bc_object = dlfn.DirichletBC(self._Vh.sub(component_index),
                                             self._null_scalar, bndry_id,
                                             "pointwise")
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.constant:
                assert isinstance(value, (tuple, list))
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(self._Vh, const_function,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.constant_component:
                assert isinstance(value, float)
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(self._Vh.sub(component_index),
                                             const_function,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.function:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(self._Vh, value,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.function_component:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(self._Vh.sub(component_index),
                                             value,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            else:  # pragma: no cover
                raise RuntimeError()
            # HINT: traction boundary conditions are covered in _setup_problem

    def _setup_function_spaces(self):
        """
        Class method setting up function spaces.
        """
        assert hasattr(self, "_mesh")
        cell = self._mesh.ufl_cell()

        assert hasattr(self, "_elastic_law")
        if self._elastic_law.compressiblity_type == "Compressible":
            # element formulation
            elemU = dlfn.VectorElement("CG", cell, self._p_deg)
            # mixed function space
            self._Vh = dlfn.FunctionSpace(self._mesh, elemU)
            self._n_dofs = self._Vh.dim()
            # print info
            assert hasattr(self, "_n_cells")
            dlfn.info("Number of cells {0}, number of DoFs: {1}".format(self._n_cells, self._n_dofs))
        elif self._elastic_law.compressiblity_type == "Incompressible":
            assert self._p_deg >= 2
            # element formulation
            elemU = dlfn.VectorElement("CG", cell, self._p_deg)
            elemP = dlfn.FiniteElement("CG", cell, self._p_deg - 1)
            mixed_element = dlfn.MixedElement([elemU, elemP])
            # mixed function space
            self._Wh = dlfn.FunctionSpace(self._mesh, mixed_element)

            self._Vh, self._Ph = self._Wh.split()
            self._n_dofs = self._Wh.dim()
            # print info
            assert hasattr(self, "_n_cells")
            dlfn.info("Number of cells {0}, number of DoFs: {1} (consisting of {2} solid DoFs and {3} pressure DoFs)".format(self._n_cells, self._n_dofs, self._Vh.dim(), self._Ph.dim()))

    def set_dimensionless_numbers(self, C, D=None):
        """
        Updates the parameters of the model by creating or modifying class
        objects.
        """
        assert isinstance(C, float)
        assert isfinite(C)
        assert C > 0.0
        if not hasattr(self, "_C"):
            self._C = dlfn.Constant(C)
        else:
            self._C.assign(C)

        if D is not None:
            assert isinstance(D, float)
            assert isfinite(D)
            assert D > 0.0
            if not hasattr(self, "_D"):
                self._D = dlfn.Constant(D)
            else:
                self._D.assign(D)

    def _setup_problem(self):
        with dlfn.Timer() as t:
            """
            Method setting up solver objects of the stationary problem.
            """
            assert hasattr(self, "_mesh")
            assert hasattr(self, "_boundary_markers")
            assert hasattr(self, "_elastic_law")

            if not hasattr(self, "_Vh"):
                self._setup_function_spaces()

            self._setup_boundary_conditions()

            # volume element
            self._dV = dlfn.Measure("dx", domain=self._mesh)
            self._dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)

            # setup the parameters for the elastic law
            self._elastic_law.set_parameters(self._mesh, self._C)

            if self._elastic_law.compressiblity_type == "Compressible" and not hasattr(self, "_solution"):
                # creating test function
                self._v = dlfn.TestFunction(self._Vh)

                # solution
                self._solution = dlfn.Function(self._Vh)

                # virtual work
                self._dw_int = self._elastic_law.dw_int(self._solution, self._v) * self._dV

            elif self._elastic_law.compressiblity_type == "Incompressible" and not hasattr(self, "_solution"):
                # creating test function
                self._v, self._q = dlfn.TestFunctions(self._Wh)

                # solution
                self._solution = dlfn.Function(self._Wh)
                self._u, self._p = dlfn.split(self._solution)

                # virtual work
                self._dw_int = self._elastic_law.dw_int(self._u, self._p, self._v, self._q) * self._dV

            # virtual work of external forces
            self._dw_ext = dlfn.dot(self._null_vector, self._v) * self._dV

            # add body force term
            if hasattr(self, "_body_force"):
                assert hasattr(self, "_D"), "Dimensionless parameter related to" + \
                                            "the body forces is not specified."
                self._dw_ext += self._D * dot(self._body_force, self._v) * self._dV

            # add boundary tractions
            if hasattr(self, "_traction_bcs"):
                for bc in self._traction_bcs:
                    # unpack values
                    if len(bc) == 3:
                        bc_type, bndry_id, traction = bc
                    elif len(bc) == 4:
                        bc_type, bndry_id, component_index, traction = bc

                    if bc_type is TractionBCType.constant:
                        assert isinstance(traction, (tuple, list))
                        const_function = dlfn.Constant(traction)
                        self._dw_ext += dot(const_function, self._v) * self._dA(bndry_id)

                    elif bc_type is TractionBCType.constant_component:
                        assert isinstance(traction, float)
                        const_function = dlfn.Constant(traction)
                        self._dw_ext += const_function * self._v[component_index] * self._dA(bndry_id)

                    elif bc_type is TractionBCType.function:
                        assert isinstance(traction, dlfn.Expression)
                        self._dw_ext += dot(traction, self._v) * self._dA(bndry_id)

                    elif bc_type is TractionBCType.function_component:
                        assert isinstance(traction, dlfn.Expression)
                        self._dw_ext += traction * self._v[component_index] * self._dA(bndry_id)

                    elif bc_type is TractionBCType.constant_pressure:
                        assert isinstance(traction, float)
                        self._dw_ext += traction * dot(self._elastic_law._normal_transform, self._v) * self._dA(bndry_id)

                    elif bc_type is TractionBCType.function_pressure:
                        assert isinstance(traction, dlfn.Expression)
                        if self._use_scaling_factor:
                            traction.scaling_factor = self._scaling_factor
                        self._dw_ext += traction * dot(self._elastic_law._normal_transform, self._v) * self._dA(bndry_id)

            self._Form = self._dw_int - self._dw_ext
            self._J_newton = dlfn.derivative(self._Form, self._solution)

            self._use_preconditioner = False
            if self._use_preconditioner:
                dlfn.info("Solving Problem with preconditioner and iterative solver.")
                A = self._elastic_law.preconditioner(self._u, self._p, self._v, self._q, self._dV)
                self._P = dlfn.derivative(A, self._solution)
                self._problem = PreconditionedNonlinearVariationalProblem(self._Form,
                                                                self._solution,
                                                                self._dirichlet_bcs,
                                                                self._J_newton,
                                                                self._P)
                # setup preconditioned nonlinear variational solver
                self._solver = PreconditionedNonlinearVariationalSolver(self._problem)
            else:
                self._problem = dlfn.NonlinearVariationalProblem(self._Form,
                                                                self._solution,
                                                                self._dirichlet_bcs,
                                                                J=self._J_newton)
                # setup nonlinear variational solver
                self._solver = dlfn.NonlinearVariationalSolver(self._problem)
            
            dlfn.info("Time to setup elastic problem: %f s" %t.elapsed()[0])

    def solve(self):
        with dlfn.Timer() as t:
            """
            Solves the elastic problem.
            """

            # setup problem

            if not all(hasattr(self, attr) for attr in ("_solver", "_problem", "_solution")) or self._use_scaling_factor:
                self._setup_problem()

            dlfn.info("Starting solution of elastic problem...")

            self._solver.solve()

            dlfn.info("Time to solve elastic problem: %f s" %t.elapsed()[0])


class PreconditionedNonlinearVariationalProblem(dlfn.NonlinearProblem):
    
    def __init__(self, Form, solution, bcs, J, P):
        self.linear_form = Form
        self.solution = solution
        self.bcs = bcs
        self.bilinear_form = J
        self.preconditioner_form = P
        dlfn.NonlinearProblem.__init__(self)

    def F(self, b, x):
        pass

    def J(self, A, x):
        pass

    def form(self, A, P, b, x):
        dlfn.assemble(self.linear_form, tensor=b)
        dlfn.assemble(self.bilinear_form, tensor=A)
        dlfn.assemble(self.preconditioner_form, tensor=P)
        for bc in self.bcs:
            bc.apply(b, x)
            bc.apply(A)
            bc.apply(P)


class PreconditionedNonlinearVariationalSolver(dlfn.NewtonSolver):

    def __init__(self, problem):
        self._problem = problem
        self._solution = self._problem.solution
        self._mesh = self._solution.function_space().mesh()
        dlfn.NewtonSolver.__init__(self, self._mesh.mpi_comm(),
                              dlfn.PETScKrylovSolver(), dlfn.PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operators(A, P)

        dlfn.PETScOptions.set("ksp_type", "gmres")
        dlfn.PETScOptions.set("pc_type", "hypre")
        #dlfn.PETScOptions.set("pc_hypre_boomeramg_cycle_type", "w")
        #dlfn.PETScOptions.set("pc_hypre_boomeramg_print_statistics", "1")
        #dlfn.PETScOptions.set("ksp_view")
        #dlfn.PETScOptions.set("ksp_monitor")

        self.linear_solver().set_from_options()

    def solve(self):
        super().solve(self._problem, self._solution.vector())