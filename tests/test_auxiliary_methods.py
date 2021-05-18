#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube
from grid_generator import HyperCubeBoundaryMarkers
from grid_generator import hyper_simplex
from grid_generator import HyperSimplexBoundaryMarkers
from auxiliary_methods import ElasticModuli
from auxiliary_methods import boundary_normal
from auxiliary_methods import compute_elasticity_coefficients
from auxiliary_methods import extract_all_boundary_markers
import math


def compare_tuples(a, b, tol=1.0e-15):
    assert isinstance(a, tuple)
    assert isinstance(b, tuple)
    assert isinstance(tol, float) and tol > 0.0
    if all(isinstance(x, int) for x in a) and all(isinstance(x, int) for x in b):
        assert all(x == y for x, y in zip(a, b))
    elif all(isinstance(x, float) for x in a) and all(isinstance(x, float) for x in b):
        assert all(abs(x-y) < tol for x, y in zip(a, b))
    else:
        raise ValueError()


def compare_dicts(a, b, tol=1.0e-15):
    assert isinstance(a, dict)
    assert isinstance(b, dict)
    assert isinstance(tol, float) and tol > 0.0
    assert all(key in b for key in a.keys())
    assert all(abs(a[key] - b[key]) < tol for key, value in a.items())


def test_boundary_normal():
    # cube: two-dimensional case
    mesh, boundary_markers = hyper_cube(2, 8)

    normal_vectors = [(-1.0, 0.0), (1.0, 0.0),
                      (0.0, -1.0), (0.0, 1.0)]
    boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                    HyperCubeBoundaryMarkers.right.value,
                    HyperCubeBoundaryMarkers.bottom.value,
                    HyperCubeBoundaryMarkers.top.value)

    for normal, bndry_id in zip(normal_vectors, boundary_ids):
        computed_normal = boundary_normal(mesh, boundary_markers, bndry_id)
        compare_tuples(normal, computed_normal)

    # cube: three-dimensional case
    mesh, boundary_markers = hyper_cube(3, 8)

    normal_vectors = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                      (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
                      (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
    boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                    HyperCubeBoundaryMarkers.right.value,
                    HyperCubeBoundaryMarkers.bottom.value,
                    HyperCubeBoundaryMarkers.top.value,
                    HyperCubeBoundaryMarkers.back.value,
                    HyperCubeBoundaryMarkers.front.value)

    for normal, bndry_id in zip(normal_vectors, boundary_ids):
        computed_normal = boundary_normal(mesh, boundary_markers, bndry_id)
        compare_tuples(normal, computed_normal)

    # simplex: two-dimensional case
    mesh, boundary_markers = hyper_simplex(2, 2)

    sqrt_2 = math.sqrt(2.0)
    normal_vectors = [(-1.0, 0.0),
                      (0.0, -1.0),
                      (1.0 / sqrt_2, 1.0 / sqrt_2)]
    boundary_ids = (HyperSimplexBoundaryMarkers.left.value,
                    HyperSimplexBoundaryMarkers.bottom.value,
                    HyperSimplexBoundaryMarkers.diagonal.value)

    for normal, bndry_id in zip(normal_vectors, boundary_ids):
        computed_normal = boundary_normal(mesh, boundary_markers, bndry_id)
        compare_tuples(normal, computed_normal, tol=5.0e-15)


def test_elasticity_coefficients():
    E = 210.0
    nu = 0.3
    lmbda = nu * E / (1. + nu) / (1. - 2. * nu)
    mu = E / 2. / (1. + nu)
    K = lmbda + 2.0 * mu / 3.0

    expected_values = {ElasticModuli.YoungsModulus: E,
                       ElasticModuli.PoissonRatio: nu,
                       ElasticModuli.ShearModulus: mu,
                       ElasticModuli.BulkModulus: K,
                       ElasticModuli.FirstLameParameter: lmbda}

    tol = 5.0e-13
    computed_values = compute_elasticity_coefficients(E=E, nu=nu)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(E=E, K=K)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(E=E, G=mu)
    compare_dicts(expected_values, computed_values, tol=tol)
    computed_values = compute_elasticity_coefficients(E=E, mu=mu)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(E=E, lmbda=lmbda)
    compare_dicts(expected_values, computed_values, tol=tol)
    computed_values = compute_elasticity_coefficients(E=E, firstLame=lmbda)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(nu=nu, K=K)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(nu=nu, G=mu)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(nu=nu, lmbda=lmbda)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(lmbda=lmbda, mu=mu)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(lmbda=lmbda, K=K)
    compare_dicts(expected_values, computed_values, tol=tol)

    computed_values = compute_elasticity_coefficients(mu=mu, K=K)
    compare_dicts(expected_values, computed_values, tol=tol)


def test_extract_all_boundary_markers():
    # cube: two-dimensional case
    mesh, boundary_markers = hyper_cube(2, 8)
    expected_boundary_ids = sorted((HyperCubeBoundaryMarkers.left.value,
                                    HyperCubeBoundaryMarkers.right.value,
                                    HyperCubeBoundaryMarkers.bottom.value,
                                    HyperCubeBoundaryMarkers.top.value))
    compare_tuples(tuple(extract_all_boundary_markers(mesh, boundary_markers)),
                   tuple(expected_boundary_ids))

    # cube: three-dimensional case
    mesh, boundary_markers = hyper_cube(3, 8)
    expected_boundary_ids = sorted((HyperCubeBoundaryMarkers.left.value,
                                    HyperCubeBoundaryMarkers.right.value,
                                    HyperCubeBoundaryMarkers.bottom.value,
                                    HyperCubeBoundaryMarkers.top.value,
                                    HyperCubeBoundaryMarkers.back.value,
                                    HyperCubeBoundaryMarkers.front.value))
    compare_tuples(tuple(extract_all_boundary_markers(mesh, boundary_markers)),
                   tuple(expected_boundary_ids))

    # simplex: two-dimensional case
    mesh, boundary_markers = hyper_simplex(2, 2)
    expected_boundary_ids = sorted((HyperSimplexBoundaryMarkers.left.value,
                                    HyperSimplexBoundaryMarkers.bottom.value,
                                    HyperSimplexBoundaryMarkers.diagonal.value))
    compare_tuples(tuple(extract_all_boundary_markers(mesh, boundary_markers)),
                   tuple(expected_boundary_ids))


if __name__ == "__main__":
    test_boundary_normal()
    test_elasticity_coefficients()
    test_extract_all_boundary_markers()