#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube
from grid_generator import spherical_shell
from grid_generator import hyper_simplex


def test_hyper_cube():
    # two-dimensional case
    _, _ = hyper_cube(2, 8)
    # three-dimensional case
    _, _ = hyper_cube(3, 8)


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


if __name__ == "__main__":
    test_hyper_cube()
    test_spherical_shell()
    test_hyper_simplex()
