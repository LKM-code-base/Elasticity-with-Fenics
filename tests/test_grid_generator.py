#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube
from grid_generator import hyper_rectangle
from grid_generator import spherical_shell
from grid_generator import hyper_simplex
from grid_generator import cylinder
from grid_generator import tire


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


def test_tire():
    # two-dimensional case
    _, _ = tire(2,"tire2D", n_refinements=0)
    # three-dimensional case
    for n_refinements in range(2):
        _, _ = tire(3,"tire3Deight", n_refinements=n_refinements)
    
    _, _ = tire(3, "tire3Dquarter")


if __name__ == "__main__":
    test_hyper_cube()
    test_hyper_rectangle()
    test_spherical_shell()
    test_hyper_simplex()
    test_cylinder()
    test_tire()
