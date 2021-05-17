#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube
from grid_generator import HyperCubeBoundaryMarkers
from grid_generator import hyper_simplex
from grid_generator import HyperSimplexBoundaryMarkers
from auxiliary_methods import boundary_normal
import math


def compare_tuples(a, b):
    assert all(abs(x-y) < 5.0e-15 for x,y in zip(a,b))


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
                      (1.0 / sqrt_2, 1.0 / sqrt_2 )]
    boundary_ids = (HyperSimplexBoundaryMarkers.left.value,
                    HyperSimplexBoundaryMarkers.bottom.value,
                    HyperSimplexBoundaryMarkers.diagonal.value)

    for normal, bndry_id in zip(normal_vectors, boundary_ids):
        computed_normal = boundary_normal(mesh, boundary_markers, bndry_id)
        compare_tuples(normal, computed_normal)


if __name__ == "__main__":
    test_boundary_normal()
