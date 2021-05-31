#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import Enum, auto

import math

import dolfin as dlfn
from mshr import Sphere, Circle, Cylinder, Polygon, generate_mesh


class GeometryType(Enum):
    spherical_annulus = auto()
    rectangle = auto()
    square = auto()
    other = auto()


class SphericalAnnulusBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a spherical annulus uniquely.
    """
    interior_boundary = auto()
    exterior_boundary = auto()


class HyperCubeBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a hyper rectangle uniquely.
    """
    left = auto()
    right = auto()
    bottom = auto()
    top = auto()
    back = auto()
    front = auto()


class HyperSimplexBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a hyper simplex uniquely.
    """
    left = auto()
    bottom = auto()
    diagonal = auto()
    # only used in 1D
    right = auto()
    # only used in 3D
    back = auto()

class CylinderBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a cylinder uniquely.
    """
    top = auto()
    bottom = auto()
    side = auto()

class CircularBoundary(dlfn.SubDomain):
    def __init__(self, **kwargs):
        super().__init__()
        assert(isinstance(kwargs["mesh"], dlfn.Mesh))
        assert(isinstance(kwargs["radius"], float) and kwargs["radius"] > 0.0)
        self._hmin = kwargs["mesh"].hmin()
        self._radius = kwargs["radius"]

    def inside(self, x, on_boundary):
        # tolerance: half length of smallest element
        tol = self._hmin / 2.
        result = abs(math.sqrt(x[0]**2 + x[1]**2) - self._radius) < tol
        return result and on_boundary

class CylinderBoundary(dlfn.SubDomain):
    def __init__(self, **kwargs):
        super().__init__()
        assert(isinstance(kwargs["mesh"], dlfn.Mesh))
        self._hmin = kwargs["mesh"].hmin()
        self._space_dim = kwargs["mesh"].geometry().dim()
        self._radius = kwargs["radius"]

    def inside(self, x, on_boundary):
        # tolerance: half length of smallest element
        tol = self._hmin / 2.
        if self._space_dim == 2:
            result = abs(math.sqrt(x[0]**2)- self._radius(x[1])) < tol
        elif self._space_dim == 3:
            result = abs(math.sqrt(x[0]**2 + x[1]**2) - self._radius(x[2])) < tol
        return result and on_boundary

def spherical_shell(dim, radii, n_refinements=0):
    """
    Creates the mesh of a spherical shell using the mshr module.
    """
    assert isinstance(dim, int)
    assert dim == 2 or dim == 3

    assert isinstance(radii, (list, tuple)) and len(radii) == 2
    ri, ro = radii
    assert isinstance(ri, float) and ri > 0.
    assert isinstance(ro, float) and ro > 0.
    assert ri < ro

    assert isinstance(n_refinements, int) and n_refinements >= 0

    # mesh generation
    if dim == 2:
        center = dlfn.Point(0., 0.)
    elif dim == 3:
        center = dlfn.Point(0., 0., 0.)

    if dim == 2:
        domain = Circle(center, ro) \
               - Circle(center, ri)
        mesh = generate_mesh(domain, 75)
    elif dim == 3:
        domain = Sphere(center, ro) \
               - Sphere(center, ri)
        mesh = generate_mesh(domain, 15)

    # mesh refinement
    for i in range(n_refinements):
        mesh = dlfn.refine(mesh)

    # MeshFunction for boundaries ids
    facet_marker = dlfn.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)

    # mark boundaries
    BoundaryMarkers = SphericalAnnulusBoundaryMarkers
    gamma_inner = CircularBoundary(mesh=mesh, radius=ri)
    gamma_inner.mark(facet_marker, BoundaryMarkers.interior_boundary.value)
    gamma_outer = CircularBoundary(mesh=mesh, radius=ro)
    gamma_outer.mark(facet_marker, BoundaryMarkers.exterior_boundary.value)

    return mesh, facet_marker


def hyper_cube(dim, n_points=10):
    assert isinstance(dim, int)
    assert dim == 2 or dim == 3
    assert isinstance(n_points, int) and n_points >= 0

    # mesh generation
    if dim == 2:
        corner_points = (dlfn.Point(0., 0.), dlfn.Point(1., 1.))
        mesh = dlfn.RectangleMesh(*corner_points, n_points, n_points)
    else:
        corner_points = (dlfn.Point(0., 0., 0.), dlfn.Point(1., 1., 1.))
        mesh = dlfn.BoxMesh(*corner_points, n_points, n_points, n_points)
    assert dim == mesh.topology().dim()

    # MeshFunction for boundaries ids
    facet_marker = dlfn.MeshFunction("size_t", mesh, dim - 1)
    facet_marker.set_all(0)

    # mark boundaries
    BoundaryMarkers = HyperCubeBoundaryMarkers
    gamma01 = dlfn.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
    gamma02 = dlfn.CompiledSubDomain("near(x[0], 1.0) && on_boundary")
    gamma03 = dlfn.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
    gamma04 = dlfn.CompiledSubDomain("near(x[1], 1.0) && on_boundary")

    gamma01.mark(facet_marker, BoundaryMarkers.left.value)
    gamma02.mark(facet_marker, BoundaryMarkers.right.value)
    gamma03.mark(facet_marker, BoundaryMarkers.bottom.value)
    gamma04.mark(facet_marker, BoundaryMarkers.top.value)

    if dim == 3:
        gamma05 = dlfn.CompiledSubDomain("near(x[2], 0.0) && on_boundary")
        gamma06 = dlfn.CompiledSubDomain("near(x[2], 1.0) && on_boundary")

        gamma05.mark(facet_marker, BoundaryMarkers.back.value)
        gamma06.mark(facet_marker, BoundaryMarkers.front.value)

    return mesh, facet_marker


def hyper_simplex(dim, n_refinements=0):
    assert isinstance(dim, int)
    assert dim <= 2, "This method is only implemented in 1D and 2D."
    assert isinstance(n_refinements, int) and n_refinements >= 0

    # mesh generation
    if dim == 1:
        mesh = dlfn.UnitIntervalMesh(n_refinements)
    elif dim == 2:
        mesh = dlfn.UnitTriangleMesh.create()

    # mesh refinement
    if dim != 1:
        for i in range(n_refinements):
            mesh = dlfn.refine(mesh)

    # MeshFunction for boundaries ids
    facet_marker = dlfn.MeshFunction("size_t", mesh, dim - 1)
    facet_marker.set_all(0)

    # mark boundaries
    BoundaryMarkers = HyperSimplexBoundaryMarkers
    if dim == 1:
        gamma01 = dlfn.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
        gamma02 = dlfn.CompiledSubDomain("near(x[0], 1.0) && on_boundary")

        gamma01.mark(facet_marker, BoundaryMarkers.left.value)
        gamma02.mark(facet_marker, BoundaryMarkers.right.value)

    elif dim == 2:
        gamma00 = dlfn.CompiledSubDomain("on_boundary")
        gamma01 = dlfn.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
        gamma02 = dlfn.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
        # first mark the entire boundary with the diagonal id
        gamma00.mark(facet_marker, BoundaryMarkers.diagonal.value)
        # then mark the other edges with the correct ids
        gamma01.mark(facet_marker, BoundaryMarkers.left.value)
        gamma02.mark(facet_marker, BoundaryMarkers.bottom.value)

    return mesh, facet_marker

def cylinder(dim, radii, height, n_refinements=0):
    """
    Creates the mesh of a cylinder using the mshr module.
    """
    assert isinstance(dim, int)
    assert dim == 2 or dim == 3

    assert isinstance(radii, (list, tuple)) and len(radii) == 2
    rt, rb = radii
    assert isinstance(rb, float) and rb > 0.
    assert isinstance(rt, float) and rt > 0.

    assert isinstance(height,float) and height > 0.

    assert isinstance(n_refinements, int) and n_refinements >= 0

    # mesh generation
    if dim == 2:
        bottom = dlfn.Point(0., 0.)
        top = dlfn.Point(0., height)
    elif dim == 3:
        bottom = dlfn.Point(0., 0., 0.)
        top = dlfn.Point(0., 0., height)

    if dim == 2:
        domain = Polygon([ dlfn.Point(-rb/2., 0.),
                           bottom,
                           dlfn.Point(rb/2., 0.),
                           dlfn.Point(rt/2., height),
                           top,
                           dlfn.Point(-rt/2.,height)])
        mesh = generate_mesh(domain, 10)
    elif dim == 3:
        domain = Cylinder(top, bottom, rt, rb, segments = 100)
        mesh = generate_mesh(domain, 32)

    # mesh refinement
    for i in range(n_refinements):
        mesh = dlfn.refine(mesh)

    # MeshFunction for boundaries ids
    facet_marker = dlfn.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)

    # calculate radius depending on the x[3] coordinate
    radius = lambda z: rb + z * (rt - rb)/height

    # mark boundaries
    BoundaryMarkers = CylinderBoundaryMarkers
    
    gamma_side = CylinderBoundary(mesh = mesh, radius = radius)
    gamma_side.mark(facet_marker, BoundaryMarkers.side.value)

    if dim == 2:
        gamma_top = dlfn.CompiledSubDomain("near(x[1], height) && on_boundary", height = height)
        gamma_top.mark(facet_marker, BoundaryMarkers.top.value)
        gamma_bottom = dlfn.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
        gamma_bottom.mark(facet_marker, BoundaryMarkers.bottom.value)
    elif dim == 3:
        gamma_top = dlfn.CompiledSubDomain("near(x[2], height) && on_boundary", height = height)
        gamma_top.mark(facet_marker, BoundaryMarkers.top.value)
        gamma_bottom = dlfn.CompiledSubDomain("near(x[2], 0.0) && on_boundary")
        gamma_bottom.mark(facet_marker, BoundaryMarkers.bottom.value)

    return mesh, facet_marker