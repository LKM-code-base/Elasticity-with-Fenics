#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum, auto
from discrete_time import DiscreteTime
import math


class BDFTimeStepping(DiscreteTime):
    """Time stepping using adaptive backward differentiation formulas for
    derivative of degree upto three.
    
    """
    def __init__(self, start_time, end_time, order=2, desired_start_time_step=0.0):
        super().__init__(start_time, end_time, desired_start_time_step)

        assert isinstance(order, int)
        assert order > 0
        self._order = order

        self._coefficients_changed = True

        # initialize time step ratio
        self._omega = -1.0

        # initialize coefficient dictionary
        self._alpha = dict()
        # coefficients of the first derivative
        self._alpha[1] = [0.0, ] * (self._order + 1)
        self._alpha[1][0] = 1.0   # initialize with values of
        self._alpha[1][1] = -1.0  # a first order scheme
        # coefficients of the second derivative
        self._alpha[2] = [0.0, ] * (self._order + 2)
        self._alpha[2][0] = 1.0   # initialize with values of
        self._alpha[2][1] = -2.0  # a first order scheme
        self._alpha[2][2] = 1.0
        # coefficients of the third derivative
        self._alpha[3] = [0.0, ] * (self._order + 3)
        self._alpha[3][0] = 1.0   # initialize with values of
        self._alpha[3][1] = -3.0  # a first order scheme
        self._alpha[3][2] = 3.0
        self._alpha[3][3] = -1.0
        # coefficients of the fourth derivative
        self._alpha[4] = [0.0, ] * (self._order + 4)
        self._alpha[4][0] = 1.0   # initialize with values of
        self._alpha[4][1] = -4.0  # a first order scheme
        self._alpha[4][2] = 6.0
        self._alpha[4][3] = -4.0
        self._alpha[4][4] = 1.0

    def restart(self):
        """
        Resets all member variables to the initial state.
        """
        super().restart()

        self._coefficients_changed = True

        # initialize time step ratio
        self._omega = -1.0

        # coefficients of the first derivative
        self._alpha[1] = [0.0, ] * (self._order + 1)
        self._alpha[1][0] = 1.0   # initialize with values of
        self._alpha[1][1] = -1.0  # a first order scheme
        # coefficients of the second derivative
        self._alpha[2] = [0.0, ] * (self._order + 2)
        self._alpha[2][0] = 1.0   # initialize with values of
        self._alpha[2][1] = -2.0  # a first order scheme
        self._alpha[2][2] = 1.0
        # coefficients of the third derivative
        self._alpha[3] = [0.0, ] * (self._order + 3)
        self._alpha[3][0] = 1.0   # initialize with values of
        self._alpha[3][1] = -3.0  # a first order scheme
        self._alpha[3][2] = 3.0
        self._alpha[3][3] = -1.0
        # coefficients of the fourth derivative
        self._alpha[4] = [0.0, ] * (self._order + 4)
        self._alpha[4][0] = 1.0   # initialize with values of
        self._alpha[4][1] = -4.0  # a first order scheme
        self._alpha[4][2] = 6.0
        self._alpha[4][3] = -4.0
        self._alpha[4][4] = 1.0

    def update_coefficients(self):
        # do not change coefficients in the first step
        if (self._step_number == 0):
            return
        # compute new time step ratio
        omega = self.get_next_step_size() / self.get_previous_step_size()
        assert math.isfinite(omega)
        assert omega > 0.0
        
        #TODO: Capture OMEGA
        Omega = omega
        #TODO: Capture XI
        Xi = Omega
        
        # check if time step ratio has changed and update the IMEX coefficients
        # from the first to the second time step the coefficients change because
        # of switching from a first order to a second order scheme
        if self._omega == omega and self._step_number > 1:
            self._coefficients_changed = False
            return
        else:
            self._omega = omega

            a, b = self._imex_parameters
            
            if self._order == 1:
                # coefficients of the first derivative
                self._alpha[1][0] = 1.0
                self._alpha[1][1] = -1.0
                # coefficients of the second derivative
                self._alpha[2][0] = 2.0 * omega / (1.0 + omega)
                self._alpha[2][1] = -2.0 * omega
                self._alpha[2][2] = 2.0 * omega * omega / (1.0 + omega)
                # coefficients of the third derivative
                num = 6.0 * omega**2 * Omega
                denom = (1.0 + omega) * (1.0 + Omega + omega * Omega)
                self._alpha[3][0] = num / denom
                num = -6.0 * omega**2 * Omega
                denom = 1.0 + Omega
                self._alpha[3][1] = num / denom
                num = 6.0 * omega**3 * Omega
                denom = 1.0 + omega
                self._alpha[3][2] = num / denom
                num = -6.0 * omega**3 * Omega**3
                denom = (1.0 + Omega) * (1.0 + Omega + omega * Omega)
                self._alpha[3][3] = num / denom
                # coefficients of the fourth derivative
                num = 24.0 * Xi * omega**3 * Omega**2
                denom = (1.0 + omega) * (1.0 + Omega + omega * Omega) *\
                    (1.0 + Xi * (1.0 + Omega + omega * Omega))
                self._alpha[4][0] = num / denom
                num = -24.0 * Xi * omega**3 * Omega**2
                denom = (1.0 + Omega) * (1.0 + Xi + Xi * Omega)
                self._alpha[4][1] = num / denom
                num = 24.0 * Xi * omega**4 * Omega**2
                denom = 1.0 + Xi + omega + Xi * omega
                self._alpha[4][2] = num / denom
                num = -24.0 * Xi * omega**4 * Omega**4
                denom = (1.0 + Omega) * (1.0 + Omega + omega * Omega)
                self._alpha[4][3] = num / denom
                num = 24.0 * Xi**4 * omega**4 * Omega**4
                denom = (1.0 + Xi) * (1.0 + Xi + Xi * Omega) *\
                    (1.0 + Xi * (1.0 + Omega + omega * Omega))
                self._alpha[4][4] = num / denom
            elif self._order == 2:
                # coefficients of the first derivative
                self._alpha[1][0] = (1.0 + 2.0 * omega) / (1.0 + omega)
                self._alpha[1][1] = -(1.0 + omega)
                self._alpha[1][2] = omega * omega / (1.0 + omega)
                # coefficients of the second derivative
                num = 2.0 * omega * (1.0 + (2.0 + 3.0 * omega) * Omega)
                denom = (1.0 + omega) * (1.0 + Omega + omega * Omega)
                self._alpha[2][0] = num / denom
                num = -2.0 * omega * (1.0 + 2.0 * (1.0 + omega) * Omega)
                denom = 1.0 + Omega
                self._alpha[2][1] = num / denom
                num = 2.0 * omega**2 * (1.0 + Omega + 2.0 * omega * Omega)
                denom = 1.0 + omega
                self._alpha[2][2] = num / denom
                num = -2.0 * omega**2 * (1.0 + 2.0 * omega) * Omega**3
                denom = (1.0 + Omega) * (1.0 + Omega + omega * Omega)
                self._alpha[2][3] = num / denom
                # coefficients of the third derivative
                num = 6.0 * omega**2 * Omega *\
                        (1.0 + Xi * (2.0 + (3.0 + 4.0 * omega) * Omega))
                denom = (1.0 + omega) * (1.0 + Omega + omega * Omega) *\
                        (1.0 + Xi * (1.0 + Omega + omega * Omega))
                self._alpha[3][0] = num / denom
                
                num = -6.0 * omega**2 * Omega *\
                        (1.0 + Xi * (2.0 + 3.0 * (1.0 + omega) * Omega))
                denom = (1.0 + Omega) * (1.0 + Xi + Xi * Omega)
                self._alpha[3][1] = num / denom
                num = 6.0 * omega**3 * Omega *\
                        (1.0 + Xi * (2.0 + (2.0 + 3.0 * omega) * Omega))
                denom = (1.0 + Xi) * (1.0 + omega)
                self._alpha[3][2] = num / denom
                num = -6.0 * omega**3 * Omega**3 *\
                        (1.0 + Xi + Xi * (2.0 + 3.0 * omega) * Omega)
                denom = (1.0 + Omega) * (1.0 + Omega + omega * Omega)
                self._alpha[3][3] = num / denom
                num = 6.0 * Xi**4 * omega**3 * Omega**3 *\
                        (1.0 + (2.0 + 3.0 * omega) * Omega)
                denom = (1.0 + Xi) * (1.0 + Xi + Xi * Omega) *\
                        (1.0 + Xi * (1.0 + Omega + omega * Omega))
                self._alpha[3][4] = num / denom
                # coefficients of the fourth derivative
                self._alpha[4] = None
            elif self._order == 3:
                # coefficients of the first derivative
                num = 1.0 + Omega + 3.0 * omega**2 * Omega + omega * (2. + 4.0 * Omega)
                denom = (1.0 + omega) * (1.0 + Omega + omega * Omega)
                self._alpha[1][0] = num / denom
                num = -(1.0 + omega) * (1.0 + Omega + omega * Omega)
                denom = (1.0 + Omega)
                self._alpha[1][1] = num / denom
                num = omega**2 * (1.0 + Omega + omega * Omega)
                denom = (1 + omega)
                self._alpha[1][2] = num / denom
                num = omega**2 * (1.0 + omega) * Omega**3
                denom = (1.0 + Omega) * (1.0 + Omega + omega * Omega)
                self._alpha[1][3] = num / denom
                # coefficients of the second derivative
                num = 2.0 * omega * (1.0 + Xi + (2.0 + 3.0 * omega) * Omega 
                                     + Xi * (4.0 + 6.0 * omega) * Omega 
                                     + Xi * (3.0 + 9.0 * omega + 6.0 * omega**2) * Omega**2)
                denom = (1.0 + omega) * (1.0 + Omega + omega * Omega) *\
                        (1.0 + Xi * (1.0 + Omega + omega * Omega))
                self._alpha[2][0] = num / denom
                
                num = -2.0 * omega * (1.0 + Xi + 2.0 * (1.0 + omega) * Omega 
                                      + 4.0 * Xi * (1.0 + omega) * Omega 
                                      + 3.0 * Xi * (1.0 + omega)**2 * Omega**2)
                denom = (1.0 + Omega) * (1.0 + Xi + Xi * Omega)
                self._alpha[2][1] = num / denom
                num = 2.0 * omega**2 * (1.0 + Xi + Omega
                                        + 2.0 * omega * Omega
                                        + Xi * (2.0 + 4.0 * omega) * Omega
                                        + Xi * (1.0 + 4.0 * omega + 3.0 * omega**2) * Omega**2)
                denom = (1.0 + Xi) * (1.0 + omega)
                self._alpha[2][2] = num / denom
                num = -2.0 * omega**2 * Omega**3 *\
                        (1.0 + 2.0 * omega + Xi * (1.0 + Omega + 3.0 * omega**2 * Omega
                                                   + omega * (2.0 + 4.0 * Omega)))
                denom = (1.0 + Omega) * (1.0 + Omega + omega * Omega)
                self._alpha[2][3] = num / denom
                num = 2.0 * Xi**4 * omega**2 * Omega**3 *\
                        (1.0 + Omega + 3.0 * omega**2 * Omega + omega * (2.0 + 4.0 * Omega))
                denom = (1.0 + Xi) * (1.0 + Xi + Xi * Omega) *\
                        (1.0 + Xi * (1.0 + Omega + omega * Omega))
                self._alpha[2][4] = num / denom
                # coefficients of the third derivative
                self._alpha[3] = None
                # coefficients of the fourth derivative
                self._alpha[4] = None
            self._coefficients_changed = True

    def print_coefficients(self):
        print("+-" + "-+-".join(4 * (12 * "-", )) + "-+")
        print("| {:12} | {:12} | {:12} | {:12} |"
              .format("coefficient", "n + 1", "n", "n - 1"))
        print("| {:12} | {:12.2e} | {:12.2e} | {:12.2e} |"
              .format("alpha", *self._alpha))
        print("| {:12} | ".format("beta") + 12 * " " +
              " | {:12.2g} | {:12.2e} |".format(*self._beta))
        print("| {:12} | {:12.2e} | {:12.2e} | {:12.2g} |"
              .format("gamma", *self._gamma))
        print("| {:12} | ".format("eta") + 12 * " " +
              " | {:12.2g} | {:12.2e} |".format(*self._eta))

    @property
    def alpha(self, derivative):
        assert derivative in (1, 2, 3, 4)
        return self._alpha[derivative]

    @property
    def coefficients_changed(self):
        return self._coefficients_changed
