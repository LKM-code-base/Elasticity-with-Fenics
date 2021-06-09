#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from ufl import inv
from dolfin import grad, inner


class ElasticLaw:
    def __init__(self):
        # book source of elastic law
        self._source = ''

    def set_parameters(self, mesh, elastic_ratio):

        assert isinstance(mesh, dlfn.Mesh)
        self._mesh = mesh

        self._space_dim = self._mesh.geometry().dim()

        self._I = dlfn.Identity(self._space_dim)

        assert isinstance(elastic_ratio, (dlfn.Constant, float))
        assert float(elastic_ratio) > 0.
        self._elastic_ratio = elastic_ratio

    def dw_int(self, u, v):
        raise NotImplementedError("You are calling a purely virtual method.")

    def postprocess_cauchy_stress(self, displacement):
        raise NotImplementedError("You are calling a purely virtual method.")

    @property
    def name(self):
        assert hasattr(self, "_name")
        return self._name

    @property
    def linearity_type(self):
        assert hasattr(self, "_linearity_type")
        return self._linearity_type

    @property
    def source(self):
        assert hasattr(self, "_source")
        return self._source


class Hooke(ElasticLaw):
    """
    Class to simulate linear elasticity with Hookes law.
    """

    def __init__(self):
        super().__init__()
        self._linearity_type = "Linear"
        self._name = "Hooke"

    def dw_int(self, u, v):
        """
        Construct internal energy.

        Parameters
        ----------
        u: TrialFunction

        v: TestFunction
        """
        # u TrialFunction (only in the linear case)
        assert isinstance(u, dlfn.function.argument.Argument)
        # v TestFunction
        assert isinstance(v, dlfn.function.argument.Argument)

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        def sym_grad(w):
            return dlfn.Constant(0.5) * (grad(w).T + grad(w))

        sigma = self._elastic_ratio * dlfn.tr(sym_grad(u)) * self._I \
            + dlfn.Constant(2.0) * sym_grad(u)

        return inner(sigma, sym_grad(v))

    def postprocess_cauchy_stress(self, displacement):
        """
        Compute Cauchy stress from given numerical solution.

        Parameters
        ----------
        displacement: Function
            Computed numerical displacement
        """
        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        assert isinstance(displacement, dlfn.function.function.Function)

        # displacement gradient
        H = grad(displacement)
        # strain tensor (symbolic)
        strain = dlfn.Constant(0.5) * (H + H.T)

        # dimensionless Cauchy stress tensor (symbolic)
        sigma = dlfn.Constant(self._elastic_ratio) \
            * inner(self._I, strain) * self._I \
            + dlfn.Constant(2.0) * strain

        return sigma


class StVenantKirchhoff(ElasticLaw):
    """
    Class to simulate nonlinear elasticity with Saint Vernant-Kirchhoff law.
    """

    def __init__(self):
        super().__init__()
        self._linearity_type = "Nonlinear"
        self._name = "StVenantKirchhoff"

    def dw_int(self, u, v):
        """
        Construct internal energy.

        Parameters
        ----------
        u: Function

        v: TestFunction
        """
        # u Function (in the nonlinear case)
        assert isinstance(u, dlfn.function.function.Function)
        # v TestFunction
        assert isinstance(v, dlfn.function.argument.Argument)

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        # deformation gradient
        F = self._I + grad(u)
        # right Cauchy-Green tensor
        C = F.T * F

        # strain
        E = dlfn.Constant(0.5) * (C - self._I)
        # 2. Piola-Kirchhoff stress
        S = self._elastic_ratio * dlfn.tr(E) * self._I \
            + dlfn.Constant(2.0) * E

        dE = dlfn.Constant(0.5) * (F.T * grad(v) + grad(v).T * F)

        return inner(S, dE)

    def postprocess_cauchy_stress(self, displacement):
        """
        Compute Cauchy stress from given numerical solution.

        Parameters
        ----------
        displacement: Function
            Computed numerical displacement
        """
        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        assert isinstance(displacement, dlfn.function.function.Function)

        # displacement gradient
        H = dlfn.grad(displacement)
        # deformation gradient
        F = self._I + H
        # right Cauchy-Green tensor
        C = F.T * F
        # volume ratio
        J = dlfn.det(F)

        # strain tensor (symbolic)
        strain = dlfn.Constant(0.5) * (C - self._I)

        # dimensionless 2. Piola-Kirchhoff stress tensor (symbolic)
        S = dlfn.Constant(self._elastic_ratio) \
            * dlfn.inner(self._I, strain) * self._I \
            + dlfn.Constant(2.0) * strain

        # dimensionless Cauchy stress tensor (symbolic)
        sigma = (F * S * F.T) / J

        return sigma


class NeoHooke(ElasticLaw):
    """
    Class to simulate nonlinear elasticity with Neo-Hooke law,
    see Holzapfel p. 247.
    """

    def __init__(self, source='Holzapfel'):
        super().__init__()
        assert isinstance(source, str) and source.lower() in ('holzapfel', 'belytschko', 'abaqus')
        self._linearity_type = "Nonlinear"
        self._name = "Neo-Hooke"
        self._source = source

    def dw_int(self, u, v):
        """
        Construct internal energy.

        Parameters
        ----------
        u: Function

        v: TestFunction
        """
        # u Function (in the nonlinear case)
        assert isinstance(u, dlfn.function.function.Function)
        # v TestFunction
        assert isinstance(v, dlfn.function.argument.Argument)

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        # deformation gradient
        F = self._I + grad(u)
        # right Cauchy-Green tensor
        C = F.T * F
        # volume ratio
        J = dlfn.det(F)

        # 2. Piola-Kirchhoff stress
        if self._source.lower() == 'holzapfel':
            S = self._I - J ** (-self._elastic_ratio) * inv(C)
        elif self._source.lower() == 'belytschko':
            S = self._I + (self._elastic_ratio * dlfn.ln(J) - dlfn.Constant(1.0)) * inv(C)
        elif self._source.lower() == 'abaqus':
            S = 1. / J ** (2. / 3.) * (self._I - dlfn.tr(C) / 3. * inv(C)) \
                + (self._elastic_constant + 2. / 3.) * J * (J - 1) * self._I

        dE = dlfn.Constant(0.5) * (F.T * grad(v) + grad(v).T * F)

        return inner(S, dE)

    def postprocess_cauchy_stress(self, displacement):
        """
        Compute Cauchy stress from given numerical solution.

        Parameters
        ----------
        displacement: Function
            Computed numerical displacement
        """
        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        assert isinstance(displacement, dlfn.function.function.Function)
        # displacement gradient
        H = grad(displacement)
        # deformation gradient
        F = self._I + H
        # right Cauchy-Green tensor
        C = F.T * F
        # volume ratio
        J = dlfn.det(F)

        # dimensionless 2. Piola-Kirchhoff stress tensor (symbolic)
        if self._source.lower() == 'holzapfel':
            S = self._I - J ** (-self._elastic_ratio) * inv(C)
        elif self._source.lower() == 'belytschko':
            S = self._I + (self._elastic_ratio * dlfn.ln(J) - dlfn.Constant(1.0)) * inv(C)
        elif self._source.lower() == 'abaqus':
            S = 1. / J ** (2. / 3.) * (self._I - dlfn.tr(C) / 3. * inv(C)) \
                + (self._elastic_constant + 2. / 3.) * J * (J - 1) * self._I

        # dimensionless Cauchy stress tensor (symbolic)
        sigma = (F * S * F.T) / J

        return sigma
