#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bdf_time_stepping import BDFTimeStepping
from dolfin import *
# BDF scheme
time_stepping = BDFTimeStepping(0.0, 1.0, order=1, desired_start_time_step=0.001)


# periodic boundary condition for the boundary pair left-right and bottom-top
class PeriodicBoundary(SubDomain):
    # define what is inside the boundary
    def inside(self, x, on_boundary):
        return (bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)
                or bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary))

    # map a coordinate x in the slave domain to a coordinate y in the master domain
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1] - 1.0


# Initialize time stepping coefficients
alpha = [Constant(0.0), Constant(0.0)]
next_step_size = Constant(0.0)


# Auxiliary method
def update_time_stepping_coefficients():
    """Update time stepping coefficients ``_alpha`` and ``_next_step_size``."""
    # Update time step
    next_step_size.assign(time_stepping.get_next_step_size())
    # Update coefficients
    coefficients = time_stepping.coefficients(derivative=1)
    for i in range(len(coefficients)):
        alpha[i].assign(coefficients[i])


# Assign initial values
update_time_stepping_coefficients()

# Create mesh and define function space
mesh = UnitSquareMesh(128, 128)
cell = mesh.ufl_cell()
element = FiniteElement("CG", cell, 1)
mixed_element = MixedElement([element, element])
V = FunctionSpace(mesh, mixed_element, constrained_domain=PeriodicBoundary())
# Define solution
solution = Function(V)
old_solution = Function(V)


# Auxiliary methods
def advance_solution():
    """Advance solution objects in time."""
    old_solution.assign(solution)


# Define initial condition

# circular pulse
# distance_string = "sqrt(pow(x[0] - x0, 2) + pow(x[1] - y0, 2))"
# function_string = "exp(-pow({0} - r0, 2) / a)".format(distance_string)
# initial_field = Expression(function_string, x0=0.5, y0=0.5, r0=0.25, a=0.01, degree=3)

# plane wave
function_string = "exp(-pow(x[1] - y0, 2) / a)"
initial_field = Expression(function_string, y0=0.5, a=0.01, degree=3)
initial_velocity = Constant(0.0)


# Project initial condition
def project_initial_conditions():
    old_u_solution, old_v_solution = old_solution.split()
    # displacement
    Vaux = FunctionSpace(mesh, V.sub(0).ufl_element(),
                         constrained_domain=PeriodicBoundary())
    projected_initial_field = project(initial_field, Vaux)
    assign(old_u_solution, projected_initial_field)
    # velocity
    Vaux = FunctionSpace(mesh, V.sub(1).ufl_element(),
                         constrained_domain=PeriodicBoundary())
    projected_initial_velocity = project(initial_velocity, Vaux)
    assign(old_v_solution, projected_initial_velocity)
    solution.assign(old_solution)


project_initial_conditions()

# Define variational problem
(u, v) = TrialFunctions(V)
(du, dv) = TestFunctions(V)
old_u_solution, old_v_solution = split(old_solution)
a = (alpha[0] / next_step_size * (u * du + v * dv) - v * du + inner(grad(u), grad(dv))) * dx
L = -alpha[1] / next_step_size * (old_v_solution * dv + old_u_solution * du) * dx
# XDMF file for saving the solution
file = XDMFFile("wave.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.parameters["rewrite_function_mesh"] = False
u_solution, v_solution = solution.split()
u_solution.rename("displacement", "")
v_solution.rename("velocity", "")
# Write initial condition
file.write(u_solution, time_stepping.current_time)
file.write(v_solution, time_stepping.current_time)
# Simple time loop
n_max_steps = 1000
while not time_stepping.is_at_end() and time_stepping.step_number < n_max_steps:
    # Set next step size
    time_stepping.set_desired_next_step_size(0.005)
    # Update coefficients
    time_stepping.update_coefficients()
    update_time_stepping_coefficients()
    # Print info
    print(time_stepping)
    # Solve problem
    solve(a == L, solution)
    # Advance time
    time_stepping.advance_time()
    advance_solution()
    # Write XDMF-files
    file.write(u_solution, time_stepping.current_time)
    file.write(v_solution, time_stepping.current_time)
print(time_stepping)
