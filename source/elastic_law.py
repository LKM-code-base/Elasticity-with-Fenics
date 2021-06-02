#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from ufl import cofac, inv
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
        self._elastic_ratio = elastic_ratio

    def dw_int(self, u, v):
        raise NotImplementedError("You are calling a purely virtual method.")
    
    def postprocess_cauchy_stress(self, displacement):
        raise NotImplementedError("You are calling a purely virtual method.")


class Hooke(ElasticLaw):
    """
    Class to simulate linear elasticity with Hookes law.
    """
    def __init__(self):
        super().__init__()
        self.linearity_type = "Linear"
        self.name = "Hooke"

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

        sym_grad = lambda w: dlfn.Constant(0.5)* (grad(w).T + grad(w))

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

        displacement_gradient = dlfn.grad(displacement)
        # strain tensor (symbolic)
        strain = dlfn.Constant(0.5) * (displacement_gradient \
            + displacement_gradient.T)
        
        # dimensionless Cauchy stress tensor (symbolic)
        sigma = dlfn.Constant(self._elastic_ratio) \
            * dlfn.inner(self._I, strain) * self._I \
                + dlfn.Constant(2.0) * strain

        return sigma


class StVenantKirchhoff(ElasticLaw):
    """
    Class to simulate nonlinear elasticity with Saint Vernant-Kirchhoff law.
    """
    def __init__(self):
        super().__init__()
        self.linearity_type = "Nonlinear"
        self.name = "StVenantKirchhoff"
        
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

        # kinematic variables
        F = self._I + grad(u)
        C = F.T * F

        # strain
        E = dlfn.Constant(0.5) * (C - self._I)
        # 2. Piola-Kirchhoff stress
        S = self._elastic_ratio * dlfn.tr(E) * self._I \
            + dlfn.Constant(2.0) * E

        dE = dlfn.Constant(0.5) * (F.T * grad(v)\
             + grad(v).T * F)

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

        displacement_gradient = dlfn.grad(displacement)

        deformation_gradient = self._I + displacement_gradient
        right_cauchy_green_tensor = deformation_gradient.T * deformation_gradient
        volume_ratio = dlfn.det(deformation_gradient)

        # strain tensor (symbolic)
        strain = dlfn.Constant(0.5) * (right_cauchy_green_tensor - self._I)
        
        # dimensionless 2. Piola-Kirchhoff stress tensor (symbolic)
        S = dlfn.Constant(self._elastic_ratio) \
            * dlfn.inner(self._I, strain) * self._I \
                + dlfn.Constant(2.0) * strain

        # dimensionless Cauchy stress tensor (symbolic)
        sigma = (deformation_gradient * S * deformation_gradient.T) / volume_ratio
        
        return sigma


class NeoHooke(ElasticLaw):
    """
    Class to simulate nonlinear elasticity with Neo-Hooke law,
    see Holzapfel p. 247.
    """
    def __init__(self):
        super().__init__()
        self.linearity_type = "Nonlinear"
        self.name = "Neo-Hooke"
        
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

        # kinematic variables
        F = self._I + grad(u)
        C = F.T * F
        J = dlfn.det(F)

        # strain
        E = dlfn.Constant(0.5) * (C - self._I)
        # 2. Piola-Kirchhoff stress
        S = self._I - J ** (-self._elastic_ratio) * inv(C)

        dE = dlfn.Constant(0.5) * (F.T * grad(v)\
             + grad(v).T * F)

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

        displacement_gradient = dlfn.grad(displacement)

        deformation_gradient = self._I + displacement_gradient
        right_cauchy_green_tensor = deformation_gradient.T * deformation_gradient
        volume_ratio = dlfn.det(deformation_gradient)

        # strain tensor (symbolic)
        strain = dlfn.Constant(0.5) * (right_cauchy_green_tensor - self._I)
        
        # dimensionless 2. Piola-Kirchhoff stress tensor (symbolic)
        S = self._I \
            - volume_ratio ** (-self._elastic_ratio) * inv(right_cauchy_green_tensor)

        # dimensionless Cauchy stress tensor (symbolic)
        sigma = (deformation_gradient * S * deformation_gradient.T) / volume_ratio
        
        return sigma