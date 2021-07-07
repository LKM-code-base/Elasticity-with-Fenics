#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from ufl import inv, cofac
from dolfin import grad, inner


class ElasticLaw:
    def __init__(self):
        pass

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
    def compressiblity_type(self):
        assert hasattr(self, "_compressiblity_type")
        return self._compressiblity_type


class Hooke(ElasticLaw):
    """
    Class to simulate linear elasticity with Hookes law.
    """

    def __init__(self):
        super().__init__()
        self._linearity_type = "Linear"
        self._name = "Hooke"
        self._compressiblity_type = "Compressible"

    def dw_int(self, u, v):
        """
        Construct internal energy.

        Parameters
        ----------
        u: Function

        v: TestFunction
        """
        # u Function
        assert isinstance(u, dlfn.function.function.Function)
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
        self._compressiblity_type = "Compressible"

    def dw_int(self, u, v):
        """
        Construct internal energy.

        Parameters
        ----------
        u: Function

        v: TestFunction
        """
        # u Function
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

    def __init__(self):
        super().__init__()
        self._linearity_type = "Nonlinear"
        self._name = "Neo-Hooke"
        self._compressiblity_type = "Compressible"

    def dw_int(self, u, v):
        """
        Construct internal energy.

        Parameters
        ----------
        u: Function

        v: TestFunction
        """
        # u Function
        assert isinstance(u, dlfn.function.function.Function)
        # v TestFunction
        assert isinstance(v, dlfn.function.argument.Argument)

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        # deformation gradient
        F = self._I + grad(u)
        # normal transform
        self._normal_transform = inv(F.T) * dlfn.FacetNormal(self._mesh)
        # right Cauchy-Green tensor
        C = F.T * F
        # volume ratio
        J = dlfn.det(F)

        # 2. Piola-Kirchhoff stress
        S = self._I - J ** (-self._elastic_ratio) * inv(C)

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
        S = self._I - J ** (-self._elastic_ratio) * inv(C)

        # dimensionless Cauchy stress tensor (symbolic)
        sigma = (F * S * F.T) / J

        return sigma


class NeoHookeIncompressible(ElasticLaw):
    """
    Class to simulate nonlinear incompressible elasticity with Neo-Hooke law,
    see Holzapfel p. 238 and p. 402.
    """

    def __init__(self):
        super().__init__()
        self._linearity_type = "Nonlinear"
        self._name = "Neo-Hooke"
        self._compressiblity_type = "Incompressible"

    def dw_int(self, u, p, v, q):
        """
        Construct internal energy.

        Parameters
        ----------
        u : Function
            Displacement.
        p : Function
            Pressure.
        v : TestFunction
            Displacement.
        q : TestFunction
            Pressure.

        Returns
        -------
        Form
            Internal energy in the variational form.
        """

        # u, p Functions
        # assert isinstance(u, dlfn.function.function.Function)
        # assert isinstance(p, dlfn.function.function.Function)
        # v, q TestFunctions
        # assert isinstance(v, dlfn.function.argument.Argument)
        # assert isinstance(q, dlfn.function.argument.Argument)

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        # deformation gradient
        F = self._I + grad(u)
        # normal transform
        self._normal_transform = cofac(F.T) * dlfn.FacetNormal(self._mesh)
        # volume ratio
        J = dlfn.det(F)
        # right Cauchy-Green tensor
        C = F.T * F
        # right Cauchy-Green tensor for isochoric deformation
        # C_iso = J ** (- 2 / 3) * C

        # 2. Piola-Kirchhoff stress
        S_vol = J * p * inv(C)
        S_iso = J ** (- 2 / 3) * self._I - 1 / 3 * J ** (- 2 / 3) * dlfn.tr(C) * inv(C)
        S = S_vol + S_iso

        dE = dlfn.Constant(0.5) * (F.T * grad(v) + grad(v).T * F)

        return inner(S, dE) + (J - 1) * q

    def postprocess_cauchy_stress(self, displacement, pressure):
        """
        Compute Cauchy stress from given numerical solution.

        Parameters
        ----------
        displacement : Function
            Displacement.
        pressure : Function
            Pressure.

        Returns
        -------
        sigma : Form
            Cauchy stress.

        """

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        # assert isinstance(displacement, dlfn.function.function.Function)
        # assert isinstance(pressure, dlfn.function.function.Function)

        # displacement gradient
        H = grad(displacement)
        # deformation gradient
        F = self._I + H
        # right Cauchy-Green tensor
        C = F.T * F
        # volume ratio
        J = dlfn.det(F)

        # 2. Piola-Kirchhoff stress
        S_vol = J * pressure * inv(C)
        S_iso = J ** (- 2 / 3) * self._I - 1 / 3 * J ** (- 2 / 3) * dlfn.tr(C) * inv(C)
        S = S_vol + S_iso

        # dimensionless Cauchy stress tensor (symbolic)
        sigma = (F * S * F.T) / J

        return sigma

    def preconditioner(self, u, p, v, q, dV):
        """
        Construct preconditioner for iterative solver.

        Parameters
        ----------
        u : Function
            Displacement.
        p : Function
            Pressure.
        v : TestFunction
            Displacement.
        q : TestFunction
            Pressure.

        Returns
        -------
        Form
            Preconditioner.
        """

        # u, p Functions
        # assert isinstance(u, dlfn.function.function.Function)
        # assert isinstance(p, dlfn.function.function.Function)
        # v, q TestFunctions
        # assert isinstance(v, dlfn.function.argument.Argument)
        # assert isinstance(q, dlfn.function.argument.Argument)

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        # deformation gradient
        F = self._I + grad(u)
        # normal transform
        self._normal_transform = cofac(F.T) * dlfn.FacetNormal(self._mesh)
        # volume ratio
        J = dlfn.det(F)
        # right Cauchy-Green tensor
        C = F.T * F
        # right Cauchy-Green tensor for isochoric deformation
        # C_iso = J ** (- 2 / 3) * C

        # Preconditioner
        A = J ** (- 2 / 3) * self._I - 1 / 3 * J ** (- 2 / 3) * dlfn.tr(C) * inv(C)

        dE = dlfn.Constant(0.5) * (F.T * grad(v) + grad(v).T * F)

        return (inner(A, dE) + p * q) * dV


class MooneyRivlinIncompressible(ElasticLaw):
    """
    Class to simulate nonlinear incompressible elasticity with Mooney-Rivlin law,
    see Holzapfel p. 238 and p. ??
    """

    def __init__(self):
        super().__init__()
        self._linearity_type = "Nonlinear"
        self._name = "Mooney-Rivlin"
        self._compressiblity_type = "Incompressible"

    def dw_int(self, u, p, v, q):
        """
        Construct internal energy.

        Parameters
        ----------
        u : Function
            Displacement.
        p : Function
            Pressure.
        v : TestFunction
            Displacement.
        q : TestFunction
            Pressure.

        Returns
        -------
        Form
            Internal energy in the variational form.
        """

        # u, p Functions
        # assert isinstance(u, dlfn.function.function.Function)
        # assert isinstance(p, dlfn.function.function.Function)
        # v, q TestFunctions
        # assert isinstance(v, dlfn.function.argument.Argument)
        # assert isinstance(q, dlfn.function.argument.Argument)

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        # deformation gradient
        F = self._I + grad(u)
        # volume ratio
        J = dlfn.det(F)
        # right Cauchy-Green tensor
        C = F.T * F
        # right Cauchy-Green tensor for isochoric deformation
        C_iso = J ** (- 2 / 3) * C

        # 2. Piola-Kirchhoff stress
        S_vol = J * p * inv(C)
        S_iso = 1. / 2. * J ** (- 2. / 3.) * (
            (1. + dlfn.tr(C_iso)) * self._I
            - 1. / 3. * (dlfn.tr(C) + dlfn.tr(C)
                         * dlfn.tr(C_iso) + inner(C, C)) * inv(C) + C
        )
        S = S_vol + S_iso

        dE = dlfn.Constant(0.5) * (F.T * grad(v) + grad(v).T * F)

        return inner(S, dE) + (J - 1) * q

    def postprocess_cauchy_stress(self, displacement, pressure):
        """
        Compute Cauchy stress from given numerical solution.

        Parameters
        ----------
        displacement : Function
            Displacement.
        pressure : Function
            Pressure.

        Returns
        -------
        sigma : Form
            Cauchy stress.

        """

        assert hasattr(self, "_elastic_ratio")
        assert hasattr(self, "_I")

        # assert isinstance(displacement, dlfn.function.function.Function)
        # assert isinstance(pressure, dlfn.function.function.Function)

        # displacement gradient
        H = grad(displacement)
        # deformation gradient
        F = self._I + H
        # volume ratio
        J = dlfn.det(F)
        # right Cauchy-Green tensor
        C = F.T * F
        # right Cauchy-Green tensor for isochoric deformation
        C_iso = J ** (- 2 / 3) * C

        # 2. Piola-Kirchhoff stress
        S_vol = J * pressure * inv(C)
        S_iso = S_iso = 1. / 2. * J ** (- 2. / 3.) * (
            (1. + dlfn.tr(C_iso)) * self._I
            - 1. / 3. * (dlfn.tr(C) + dlfn.tr(C)
                         * dlfn.tr(C_iso) + inner(C, C)) * inv(C) + C
        )
        S = S_vol + S_iso

        # dimensionless Cauchy stress tensor (symbolic)
        sigma = (F * S * F.T) / J

        return sigma
