#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
from grid_generator import hyper_cube
from grid_generator import hyper_rectangle
from grid_generator import spherical_shell
from grid_generator import hyper_simplex
from grid_generator import cylinder
from grid_generator import _extract_facet_markers
from os import path
import subprocess


def test_hyper_cube():
    # two-dimensional case
    _, _ = hyper_cube(2, 8)
    # three-dimensional case
    _, _ = hyper_cube(3, 8)


def test_hyper_rectangle():
    # two-dimensional case
    _, _ = hyper_rectangle((0.0, 0.0), (10.0, 1.0), 10)
    _, _ = hyper_rectangle((0.0, 0.0), (10.0, 1.0), (50, 5))
    # three-dimensional case
    _, _ = hyper_rectangle((0.0, 0.0, 0.0), (10.0, 1.0, 2.0), 8)
    _, _ = hyper_rectangle((0.0, 0.0, 0.0), (10.0, 1.0, 2.0), (50, 5, 10))


def test_spherical_shell():
    # two-dimensional case
    _, _ = spherical_shell(2, (0.3, 1.0), 2)
    # three-dimensional case
    _, _ = spherical_shell(3, (0.3, 1.0), 2)


def test_hyper_simplex():
    # one-dimensional case
    _, _ = hyper_simplex(1, 2)
    # two-dimensional case
    _, _ = hyper_simplex(2, 2)


def test_cylinder():
    # two-dimensional case
    _, _ = cylinder(2, (0.3, 1.0), 3.0)
    # three-dimensional case
    _, _ = cylinder(3, (0.3, 1.0), 3.0)


def test_extract_boundary_markers():
    url_str = "https://github.com/LKM-code-base/Gmsh-collection/blob/66b29ba984ed6792f56666ee8eebc458c7a626d4/meshes/CubeThreeMaterials.geo"
    subprocess.run(["wget", url_str], check=True)
    fname = "CubeThreeMaterials.geo"
    geo_files = glob.glob("*.geo", recursive=True)
    for file in geo_files:
        if fname in file:
            geo_file = file
            break
    assert path.exists(geo_file)
    _ = _extract_facet_markers(geo_file)
    subprocess.run(["rm", geo_file], check=True)


if __name__ == "__main__":
    test_hyper_cube()
    test_hyper_rectangle()
    test_spherical_shell()
    test_hyper_simplex()
    test_cylinder()
    test_extract_boundary_markers()
