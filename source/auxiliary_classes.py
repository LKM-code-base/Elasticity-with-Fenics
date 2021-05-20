#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dolfin as dlfn
import numpy as np


class PointSubDomain(dlfn.SubDomain):
    def __init__(self, coord, tol = 1e-12):
        # input check
        assert isinstance(coord, (tuple, list))
        assert len(coord) > 1
        assert isinstance(tol, float)
        assert tol > 0.0
        # call constructor of parent class
        dlfn.SubDomain.__init__(self)
        # assign member variables
        self._coord = np.array(coord)
        self._tol = tol
        self._space_dim = self._coord.size

    def inside(self, x, on_boundary):
        return np.linalg.norm(x - self._coord) < self._tol