#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bdf_time_stepping import BDFTimeStepping


def compare_lists(a, b):
    assert a == b, "The list {0} is not equal to the list {1}".format(a, b)


step_sizes = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]
coefficients_changed = [True, True, True, True, True, True, False]
assert len(coefficients_changed) == len(step_sizes)


def time_loop(time_stepping, alpha):
    # simple time loop
    while not time_stepping.is_at_end():
        # extract step number and step size
        step_number = time_stepping.step_number
        step_size = step_sizes[step_number]
        # set next step size
        time_stepping.set_desired_next_step_size(step_size)
        # update coefficients
        time_stepping.update_coefficients()
        # print info
        print(time_stepping)
        time_stepping.print_coefficients()
        # check correctness of coefficients
        compare_lists(time_stepping.alpha[1], alpha[1][step_number])
        compare_lists(time_stepping.alpha[2], alpha[2][step_number])
        assert (time_stepping.coefficients_changed == coefficients_changed[step_number])
        # advance time
        time_stepping.advance_time()
    print(time_stepping)
    assert time_stepping.is_at_end()
    # restart
    time_stepping.restart()
    while not time_stepping.is_at_end():
        # extract step number and step size
        step_number = time_stepping.step_number
        step_size = step_sizes[step_number]
        # set next step size
        time_stepping.set_desired_next_step_size(step_size)
        # update coefficients
        time_stepping.update_coefficients()
        # print info
        print(time_stepping)
        time_stepping.print_coefficients()
        # check correctness of coefficients
        compare_lists(time_stepping.alpha, alpha[step_number])
        assert (time_stepping.coefficients_changed == coefficients_changed[step_number])
        # advance time
        time_stepping.advance_time()
    print(time_stepping)
    assert time_stepping.is_at_end()


def test_order_one():
    time_stepping = BDFTimeStepping(0.0, 9.0, order=1)
    alpha = [[1.0, -1.0, 0.0],
             [1.5, -2.0, 0.5],
             [5.0/3.0, -3.0, 4.0/3.0],
             [1.5, -2.0, 0.5],
             [4.0/3.0, -1.5, 1.0/6.0],
             [1.5, -2.0, 0.5],
             [1.5, -2.0, 0.5]]
    assert len(alpha) == len(step_sizes)

    time_loop(time_stepping, alpha)


if __name__ == "__main__":
    test_order_one()
