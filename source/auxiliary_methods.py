#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from math import isfinite, sqrt
from enum import Enum, auto

__all__ = ["boundary_normal", "compute_elasticity_coefficients"]


class ElasticModuli(Enum):
    YoungsModulus = auto()
    PoissonRatio = auto()
    ShearModulus = auto()
    BulkModulus = auto()
    FirstLameParameter = auto()
    SecondLameParameter = ShearModulus


def boundary_normal(mesh, facet_markers, bndry_id):
    """
    Extracts the normal vector of the boundary marked by the boundary id
    by checking that
        1. the facet normal vectors are co-linear
        2. the vector connecting two face midpoints is tangential to both
           normal vectors.
    Returns a tuple of float representing the normal.
    """
    assert isinstance(mesh, dlfn.Mesh)
    assert isinstance(facet_markers, dlfn.cpp.mesh.MeshFunctionSizet)
    assert isinstance(bndry_id, int)

    tol = 1.0e3 * dlfn.DOLFIN_EPS
    normal_vectors = []
    midpoints = []
    for f in dlfn.facets(mesh):
        if f.exterior():
            if facet_markers[f] == bndry_id:
                current_normal = f.normal()
                current_midpoint = f.midpoint()
                for normal, midpoint in zip(normal_vectors, midpoints):
                    # check that normal vectors point in the same direction
                    assert current_normal.dot(normal) > 0.0
                    # check that normal vector are parallel
                    if abs(current_normal.dot(normal) - 1.0) > tol:  # pragma: no cover
                        raise ValueError("Boundary facets do not share common normal.")
                    # compute a tangential vector as connection vector of two
                    # midpoints
                    midpoint_connection = midpoint - current_midpoint
                    # check that tangential vector is orthogonal to both normal
                    #  vectors
                    if abs(midpoint_connection.dot(normal)) > tol:  # pragma: no cover
                        raise ValueError("Midpoint connection vector is not tangential to boundary facets.")
                    if abs(midpoint_connection.dot(current_normal)) > tol:  # pragma: no cover
                        raise ValueError("Midpoint connection vector is not tangential to boundary facets.")
                normal_vectors.append(current_normal)
                midpoints.append(current_midpoint)

    assert len(normal_vectors) > 0, "Boundary id is not marked in MeshFunction"
    assert len(midpoints) == len(normal_vectors)

    dim = mesh.topology().dim()
    normal = normal_vectors[0]

    return tuple(normal[d] for d in range(dim))


def compute_elasticity_coefficients(**kwargs):
    """
    Compute various elasticity coefficients from the input dictionary.
    For example,

        compute_elasticity_coefficients(E=210.0, nu=0.3)

    returns

        {BulkModulus: ,
         ShearModulus: ,
         YoungsModulus: , PoissonsRatio: ,
         FirstLameParameter: ,
         SecondLameParameter: }
    """
    assert len(kwargs) == 2, "Only two elastic material parameters are independent"
    youngs_modulus = None
    poissons_ratio = None
    shear_modulus = None
    bulk_modulus = None
    first_lame = None

    return_dict = dict()

    for key, value in kwargs.items():
        assert isinstance(value, float)
        assert isfinite(value)
        if key == "E":
            assert value > 0.0
            youngs_modulus = value
            return_dict[ElasticModuli.YoungsModulus] = youngs_modulus

        elif key == "nu":
            assert value < 0.5
            poissons_ratio = value
            return_dict[ElasticModuli.PoissonRatio] = poissons_ratio

        elif key == "K":
            assert value > 0.0
            bulk_modulus = value
            return_dict[ElasticModuli.BulkModulus] = bulk_modulus

        elif key == "G" or key == "mu":
            assert value > 0.0
            shear_modulus = value
            return_dict[ElasticModuli.ShearModulus] = shear_modulus
            return_dict[ElasticModuli.SecondLameParameter] = shear_modulus

        elif key == "lmbda" or key.lower() == "firstlame":
            first_lame = value
            return_dict[ElasticModuli.FirstLameParameter] = first_lame

        else:  # pragma: no cover
            raise ValueError("Unexpected value key in function input.")

    if youngs_modulus is not None and poissons_ratio is not None:
        shear_modulus = youngs_modulus
        shear_modulus /= 2. * (1. + poissons_ratio)
        return_dict[ElasticModuli.ShearModulus] = shear_modulus

        bulk_modulus = youngs_modulus
        bulk_modulus /= 3.0 * (1. - 2. * poissons_ratio)
        return_dict[ElasticModuli.BulkModulus] = bulk_modulus

        first_lame = youngs_modulus * poissons_ratio
        first_lame /= ((1. + poissons_ratio) * (1. - 2. * poissons_ratio))
        return_dict[ElasticModuli.FirstLameParameter] = first_lame

    elif youngs_modulus is not None and shear_modulus is not None:
        poissons_ratio = youngs_modulus / 2. / shear_modulus - 1.0
        return_dict[ElasticModuli.PoissonRatio] = poissons_ratio

        bulk_modulus = youngs_modulus * shear_modulus
        bulk_modulus /= 3.0 * (3. * shear_modulus - youngs_modulus)
        return_dict[ElasticModuli.BulkModulus] = bulk_modulus

        first_lame = shear_modulus * (youngs_modulus - 2. * shear_modulus)
        first_lame /= 3. * shear_modulus - youngs_modulus
        return_dict[ElasticModuli.FirstLameParameter] = first_lame

    elif youngs_modulus is not None and bulk_modulus is not None:
        poissons_ratio = 3. * bulk_modulus - youngs_modulus
        poissons_ratio /= 6. * bulk_modulus
        return_dict[ElasticModuli.PoissonRatio] = poissons_ratio

        shear_modulus = 3. * bulk_modulus * youngs_modulus
        shear_modulus /= 9. * bulk_modulus - youngs_modulus
        return_dict[ElasticModuli.ShearModulus] = shear_modulus

        first_lame = 3. * bulk_modulus * (3. * bulk_modulus - youngs_modulus)
        first_lame /= 9. * bulk_modulus - youngs_modulus
        return_dict[ElasticModuli.FirstLameParameter] = first_lame

    elif youngs_modulus is not None and first_lame is not None:
        R = sqrt(youngs_modulus**2 + 9. * first_lame**2
                 + 2.0 * youngs_modulus * first_lame)
        poissons_ratio = 2.0 * first_lame
        poissons_ratio /= youngs_modulus + first_lame + R
        return_dict[ElasticModuli.PoissonRatio] = poissons_ratio

        shear_modulus = (youngs_modulus - 3. * first_lame + R) / 4.0
        return_dict[ElasticModuli.ShearModulus] = shear_modulus

        bulk_modulus = (youngs_modulus + 3. * first_lame + R) / 6.0
        return_dict[ElasticModuli.BulkModulus] = bulk_modulus

    elif poissons_ratio is not None and shear_modulus is not None:
        youngs_modulus = 2.0 * shear_modulus * (1.0 + poissons_ratio)
        return_dict[ElasticModuli.YoungsModulus] = youngs_modulus

        bulk_modulus = 2.0 * shear_modulus * (1.0 + poissons_ratio)
        bulk_modulus /= 3.0 * (1. - 2. * poissons_ratio)
        return_dict[ElasticModuli.BulkModulus] = bulk_modulus

        first_lame = 2.0 * shear_modulus * poissons_ratio
        first_lame /= (1. - 2. * poissons_ratio)
        return_dict[ElasticModuli.FirstLameParameter] = first_lame

    elif poissons_ratio is not None and bulk_modulus is not None:
        youngs_modulus = 3.0 * bulk_modulus * (1.0 - 2.0 * poissons_ratio)
        return_dict[ElasticModuli.YoungsModulus] = youngs_modulus

        shear_modulus = 3.0 * bulk_modulus * (1.0 - 2.0 * poissons_ratio)
        shear_modulus /= 2.0 * (1.0 + poissons_ratio)
        return_dict[ElasticModuli.ShearModulus] = shear_modulus

        first_lame = 3.0 * bulk_modulus * poissons_ratio
        first_lame /= 1. + poissons_ratio
        return_dict[ElasticModuli.FirstLameParameter] = first_lame

    elif poissons_ratio is not None and first_lame is not None:
        youngs_modulus = first_lame * (1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio)
        youngs_modulus /= poissons_ratio
        return_dict[ElasticModuli.YoungsModulus] = youngs_modulus

        shear_modulus = first_lame * (1.0 - 2.0 * poissons_ratio)
        shear_modulus /= 2.0 * poissons_ratio
        return_dict[ElasticModuli.ShearModulus] = shear_modulus

        bulk_modulus = first_lame * (1.0 + poissons_ratio)
        bulk_modulus /= 3.0 * poissons_ratio
        return_dict[ElasticModuli.BulkModulus] = bulk_modulus

    elif shear_modulus is not None and bulk_modulus is not None:
        youngs_modulus = 9.0 * bulk_modulus * shear_modulus
        youngs_modulus /= 3.0 * bulk_modulus + shear_modulus
        return_dict[ElasticModuli.YoungsModulus] = youngs_modulus

        poissons_ratio = 3.0 * bulk_modulus - 2.0 * shear_modulus
        poissons_ratio /= 2.0 * (3.0 * bulk_modulus + shear_modulus)
        return_dict[ElasticModuli.PoissonRatio] = poissons_ratio

        first_lame = bulk_modulus - 2.0 * shear_modulus / 3.0
        return_dict[ElasticModuli.FirstLameParameter] = first_lame

    elif shear_modulus is not None and first_lame is not None:
        youngs_modulus = shear_modulus * (3.0 * first_lame + 2.0 * shear_modulus)
        youngs_modulus /= shear_modulus + first_lame
        return_dict[ElasticModuli.YoungsModulus] = youngs_modulus

        poissons_ratio = first_lame / 2.0 / (first_lame + shear_modulus)
        return_dict[ElasticModuli.PoissonRatio] = poissons_ratio

        bulk_modulus = first_lame + 2.0 * shear_modulus / 3.0
        return_dict[ElasticModuli.BulkModulus] = bulk_modulus

    elif bulk_modulus is not None and first_lame is not None:
        youngs_modulus = 9.0 * bulk_modulus * (bulk_modulus - first_lame)
        youngs_modulus /= 3.0 * bulk_modulus - first_lame
        return_dict[ElasticModuli.YoungsModulus] = youngs_modulus

        poissons_ratio = first_lame / (3.0 * bulk_modulus - first_lame)
        return_dict[ElasticModuli.PoissonRatio] = poissons_ratio

        shear_modulus = 3.0 * (bulk_modulus - first_lame) / 2.0
        return_dict[ElasticModuli.ShearModulus] = shear_modulus
    else:  # pragma: no cover
        raise RuntimeError()

    assert len(return_dict) == 5
    assert all(isinstance(x, float) for x in return_dict.values())
    assert all(isfinite(x) for x in return_dict.values())

    return return_dict


def extract_all_boundary_markers(mesh, mesh_function):
    """
    Stores all boundary markers of the MeshFunction inside a set.
    """
    assert isinstance(mesh, dlfn.Mesh)
    assert isinstance(mesh_function, (dlfn.cpp.mesh.MeshFunctionSizet,
                                      dlfn.cpp.mesh.MeshFunctionInt))
    boundary_markers = set()
    for f in dlfn.facets(mesh):
        if f.exterior():
            boundary_markers.add(mesh_function[f])
    return boundary_markers
