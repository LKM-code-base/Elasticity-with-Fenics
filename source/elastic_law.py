#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from ufl import cofac
from dolfin import grad, inner

class ElasticLaw:
    def __init__(self):
        pass

    def set_parameters(self, mesh, C, u, v, solution):
        
        self._mesh = mesh
        self._C = C

        self._u = u
        self._v = v
        self._solution = solution

        self._space_dim = self._mesh.geometry().dim()
        self._N  = dlfn.FacetNormal(self._mesh)

        self._dV = dlfn.Measure("dx", domain=self._mesh)

        self._I = dlfn.Identity(self._space_dim)
        
    def _strain(self):
        raise NotImplementedError("You are calling a purely virtual method.")
    
    def _dstrain(self):
        raise NotImplementedError("You are calling a purely virtual method.")

    def volume_scaling(self):
        raise NotImplementedError("You are calling a purely virtual method.")

    def traction_scaling(self, bndry_id):
        raise NotImplementedError("You are calling a purely virtual method.")

    def dw_int(self):
        raise NotImplementedError("You are calling a purely virtual method.")
    
    def postprocess_cauchy_stress(self, displacement):
        raise NotImplementedError("You are calling a purely virtual method.")


class Hooke(ElasticLaw):
    def __init__(self):
        super().__init__()
        self._nonlinear = False
        self._linearity_type = "linear"
        self._name = "Hooke"

    def set_parameters(self, mesh, C, u, v, solution):
        super().set_parameters(mesh, C, u, v, solution)
        
    def _cauchy_stress(self):

        assert hasattr(self, "_C")
        assert hasattr(self, "_I")

        return self._C * dlfn.tr(self._strain()) * self._I \
            + dlfn.Constant(2.0) * self._strain()

    def _strain(self):

        assert hasattr(self, "_u")

        return dlfn.Constant(0.5) * (grad(self._u).T + grad(self._u))

    def _dstrain(self):

        assert hasattr(self, "_v")

        return dlfn.Constant(0.5) * (grad(self._v).T + grad(self._v))

    def volume_scaling(self):
        return dlfn.Constant(1.0)

    def traction_scaling(self, bndry_id):
        return dlfn.Constant(1.0)

    def dw_int(self):
        
        assert hasattr(self, "_dV")

        return inner(self._cauchy_stress(),self._dstrain()) * self._dV

    def postprocess_cauchy_stress(self, displacement):

        assert hasattr(self, "_C")
        assert hasattr(self, "_I")

        displacement_gradient = dlfn.grad(displacement)
        # strain tensor (symbolic)
        strain = dlfn.Constant(0.5) * (displacement_gradient + \
            displacement_gradient.T)
        
        # dimensionless stress tensor (symbolic)
        stress = dlfn.Constant(self._C) * \
            dlfn.inner(self._I, strain) * self._I
        stress += dlfn.Constant(2.0) * strain

        return stress


class StVenantKirchhoff(ElasticLaw):
    def __init__(self):
        super().__init__()
        self._nonlinear = True
        self._linearity_type = "nonlinear"
        self._name = "StVenantKirchhoff"
        
    def set_parameters(self, mesh, C, u, v, solution):
        super().set_parameters(mesh, C, u, v, solution)
        
        # kinematic variables
        self._F = self._I + grad(self._solution)
        self._CG = self._F.T * self._F
        self._J = dlfn.det(self._F)

    def _second_piola_kirchhoff_stress(self):

        assert hasattr(self, "_C")
        assert hasattr(self, "_I")

        return self._C * dlfn.tr(self._strain()) * self._I + \
            dlfn.Constant(2.0) * self._strain()

    def _strain(self):

        assert hasattr(self, "_I")
        assert hasattr(self, "_CG")

        return dlfn.Constant(0.5) * (self._CG - self._I)

    def _dstrain(self):

        assert hasattr(self, "_F")
        assert hasattr(self, "_v")

        return dlfn.Constant(0.5) * (self._F.T * grad(self._v) + grad(self._v).T * self._F)

    def volume_scaling(self):

        assert hasattr(self, "_J")

        return self._J

    def traction_scaling(self, bndry_id):

        assert hasattr(self, "_F")
        assert hasattr(self, "_N")

        return dlfn.sqrt(dlfn.dot(cofac(self._F) * self._N(bndry_id),\
            cofac(self._F) * self._N(bndry_id)))

    def dw_int(self):

        assert hasattr(self, "_dV")

        return inner(self._second_piola_kirchhoff_stress(),self._dstrain()) * self._dV

    def postprocess_cauchy_stress(self, displacement):

        assert hasattr(self, "_C")
        assert hasattr(self, "_I")

        displacement_gradient = dlfn.grad(displacement)
        
        # strain tensor (symbolic)
        deformation_gradient = self._I + displacement_gradient
        right_cauchy_green_tensor = deformation_gradient.T * deformation_gradient
        volume_ratio = dlfn.det(deformation_gradient)

        strain = dlfn.Constant(0.5) * (right_cauchy_green_tensor - self._I)
        # dimensionless 2. PK stress tensor (symbolic)
        S = dlfn.Constant(self._C) * dlfn.inner(self._I, strain) * self._I
        S += dlfn.Constant(2.0) * strain

        stress = (deformation_gradient * S * deformation_gradient.T) / volume_ratio

        return stress