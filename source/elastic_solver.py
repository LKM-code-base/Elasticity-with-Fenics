#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import isfinite
from enum import Enum, auto

import dolfin as dlfn
from dolfin import grad, div, dot, inner


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


class LinearElasticitySolver():
    """
    Class to simulate linear elasticity.
    """
    # class variables
    _sub_space_association = {0: "displacement"}
    _field_association = {value: key for key, value in _sub_space_association.items()}
    _apply_body_force = False
    _body_force_specified = False
    _apply_boundary_traction = False
    _null_scalar = dlfn.Constant(0.)

    def __init__(self, mesh, boundary_markers):
        # input check
        assert isinstance(mesh, dlfn.Mesh)
        assert isinstance(boundary_markers, (dlfn.cpp.mesh.MeshFunctionSizet,
                                             dlfn.cpp.mesh.MeshFunctionInt))

        # set mesh variables
        self._mesh = mesh
        self._boundary_markers = boundary_markers
        self._space_dim = self._mesh.geometry().dim()
        assert self._boundary_markers.dim() == self._space_dim - 1
        self._n_cells = self._mesh.num_cells()

        # dimension-dependent variables
        self._null_vector = dlfn.Constant((0., ) * self._space_dim)

        # set discretization parameters
        # polynomial degree
        self._p_deg = 1
        # quadrature degree
        q_deg = self._p_deg + 2
        dlfn.parameters["form_compiler"]["quadrature_degree"] = q_deg

    def _setup_function_spaces(self):
        """
        Class method setting up function spaces.
        """
        assert hasattr(self, "_mesh")
        cell = self._mesh.ufl_cell()
        # element formulation
        elemU = dlfn.VectorElement("CG", cell, self._p_deg)
        # mixed function space
        self._Vh = dlfn.FunctionSpace(self._mesh, elemU)
        self._n_dofs = self._Wh.dim()
        # print info
        assert hasattr(self, "_n_cells")
        dlfn.info("Number of cells {0}, number of DoFs: {1}".format(self._n_cells, self._n_dofs))

    def _setup_boundary_conditions(self):
        assert hasattr(self, "_bcs")
        assert hasattr(self, "_Vh")
        assert hasattr(self, "_boundary_markers")
        # dirichlet bcs
        self._dirichlet_bcs = []
        # displacement part
        disp_bcs = self._bcs["displacement"]
        for bc_type, bc_bndry_id, bc_value in disp_bcs:
            if bc_type is DisplacementBCType.fixed:
                bc_object = dlfn.DirichletBC(self._Vh, self._null_vector,
                                             self._boundary_markers, bc_bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.function:
                # TODO Check shape and type of variable bc_value
                bc_object = dlfn.DirichletBC(self._Vh, bc_value,
                                             self._boundary_markers, bc_bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is DisplacementBCType.constant:
                # TODO Check shape and type of variable bc_value
                const_function = dlfn.Constant(bc_value)
                bc_object = dlfn.DirichletBC(self._Vh, const_function,
                                             self._boundary_markers, bc_bndry_id)
                self._dirichlet_bcs.append(bc_object)

            else:  # pragma: no cover
                # TODO Implement other types of boundary conditions
                raise RuntimeError()

        # traction boundary conditions
        if "traction" in self._bcs:
            self._traction_bcs = dict()
            traction_bcs = self._bcs["traction"]
            for bc_type, bc_bndry_id, bc_value in traction_bcs:

                for _, velocity_bndry_id, _ in disp_bcs:
                    assert velocity_bndry_id != bc_bndry_id, \
                        ValueError("Unconsistent boundary conditions on "
                                   "boundary with boundary id: {0}."
                                   .format(bc_bndry_id))

                if bc_type is not TractionBCType.free:
                    # make sure that there is no velocity boundary condition on
                    # the current boundary
                    if bc_type is TractionBCType.constant:
                        const_function = dlfn.Constant(bc_value)
                        self._traction_bs[bc_bndry_id] = const_function
                    elif bc_type is TractionBCType.function:
                        self._traction_bs[bc_bndry_id] = bc_value
                    else:
                        raise NotImplementedError()

    def _setup_problem(self):
        """
        Method setting up non-linear solver objects of the stationary problem.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")

        self._setup_function_spaces()
        self._setup_boundary_conditions()

        # creating test and trial functions
        u = dlfn.TrialFunction(self._Vh)
        v = dlfn.TestFunction(self._Vh)

        # solution
        self._solution = dlfn.Function(self._Vh)

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)

        # dimensionless parameters
        C = self._C
        # weak forms
        # virtual work of internal forces
        strain = dlfn.Constant(0.5) * (grad(u) + grad(u).T)
        dw_int = ( C * div(u) * div(v) 
                 + inner(dlfn.Constant(2.0) * strain, grad(u) + grad(u).T)
                 ) * dV

        # virtual work of external forces
        dw_ext = dlfn.dot(self._null_vector, v) * dV
        # add body force term
        if self._apply_body_force is True:
            assert hasattr(self, "_body_force"), "Body force is not specified."
            assert hasattr(self, "_D"), "Dimensionless parameter related to" + \
                                        "the body forces is not specified."
            dw_ext += self._D * dot(self._body_force, v) * dV
        # add boundary tractions
        if hasattr(self, "_traction_bcs"):
            for bndry_id, traction in self._traction_bcs.items():
                dw_ext -= dot(traction, v) * dA(bndry_id)
        # linear variational problem
        self._linear_problem = dlfn.LinearVariationalProblem(dw_int, dw_ext,
                                                             self._solution,
                                                             self._dirichlet_bcs)
        # setup linear variational solver
        self._linear_solver = dlfn.LinearVariationalSolver(self._linear_problem)

    def set_boundary_conditions(self, bcs):
        """
        Set the boundary conditions of the problem.
        """
        assert isinstance(bcs, dict)
        # create a set containing contrained boundaries
        bndry_ids = set()
        # check that boundary ids passed in the bcs dictionary 
        # occur in the facet markers
        bndry_ids_found = dict(zip(bndry_ids, (False, ) * len(bndry_ids)))
        for facet in dlfn.facets(self._mesh):
            if facet.exterior():
                if self._boundary_markers[facet] in bndry_ids:
                    bndry_ids_found[self._boundary_markers[facet]] = True
                    if all(bndry_ids_found.values()):
                        break
        if not all(bndry_ids_found):
            missing = [key for key, value in bndry_ids_found.items() if value is False]
            message = "Boundary id" + ("s " if len(missing) > 1 else " ")
            message += ", ".join(map(str, missing))
            message += "were not found in the facet markers of the mesh."
            raise ValueError(message)
        # check that at least one displacement bc is specified
        assert "displacement" in bcs
        assert len(bcs["displacement"]) > 0
        # check that there is no conflict between displacement between
        # displacement and traction bcs
        if "traction" in bcs:
            displacement_bcs = bcs["displacement"]
            traction_bcs = bcs["traction"]
            # extract boundary ids with displacement bcs
            disp_bcs_bndry_ids = set()
            for bc in displacement_bcs:
                assert isinstance(bc[1], int)
                disp_bcs_bndry_ids.add(bc[1])
            # extract boundary ids with displacement bcs
            traction_bcs_bndry_ids = set()
            for bc in traction_bcs:
                assert isinstance(bc[1], int)
                traction_bcs_bndry_ids.add(bc[1])
            # compute boundary ids with simultaneous bcs
            joint_bndry_ids = disp_bcs_bndry_ids.intersection(traction_bcs_bndry_ids)
            # make sure that bcs are only applied component-wise 
            allowedDisplacementBCTypes = (DisplacementBCType.fixed_component,
                                          DisplacementBCType.fixed_component_pointwise,
                                          DisplacementBCType.constant_component,
                                          DisplacementBCType.function_component)
            allowedTractionBCTypes = (TractionBCType.constant_component,
                                      TractionBCType.function_component)
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
        self._bcs = bcs

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

    @property
    def sub_space_association(self):
        return self._sub_space_association

    @property
    def field_association(self):
        return self._field_association

    @property
    def solution(self):
        return self._solution

    def set_body_force(self, body_force):
        """
        Specifies the body force.

        Parameters
        ----------
        body_force : dolfin.Expression, dolfin. Constant
            The body force.
        """
        assert isinstance(body_force, (dlfn.Expression, dlfn.Constant))
        assert body_force.ufl_shape[0] == self._space_dim
        self._body_force = body_force
        self._body_force_specified = True

    def solve(self):
        """
        Solves the linear problem.
        """
        # setup problem
        if not all(hasattr(self, attr) for attr in ("_linear_solver",
                                                    "_linear_problem",
                                                    "_solution")):
            self._setup_problem()

        dlfn.info("Starting solution of linear elastic problem...")
        self._linear_solver.solve()
