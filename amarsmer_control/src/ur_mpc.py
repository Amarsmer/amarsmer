#!/usr/bin/env python3

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the nonlinear underwater robot model
def export_underwater_model(robot_mass = 10, 
                            iz = 5, 
                            ):
    model = AcadosModel()
    model.name = "ur_robot_model"

    # Define symbolic states
    x = ca.SX.sym('x')       # Position in x (inertial frame)
    y = ca.SX.sym('y')       # Position in y (inertial frame)
    psi = ca.SX.sym('psi')   # Yaw angle
    u = ca.SX.sym('u')       # Surge velocity
    r = ca.SX.sym('r')       # Yaw rate

    states = ca.vertcat(x, y, psi, u, r)

    # Define control inputs (thrusts)
    tau_u = ca.SX.sym('tau_u')   # Surge thrust
    tau_r = ca.SX.sym('tau_r')   # Yaw torque

    controls = ca.vertcat(tau_u, tau_r)

    # Define physical parameters
    m = robot_mass    # mass of the robot
    I_z = iz    # moment of inertia about the z-axis

    # Define the nonlinear dynamics
    x_dot = u * ca.cos(psi)      # Velocity in x
    y_dot = u * ca.sin(psi)      # Velocity in y
    psi_dot = r                  # Change in yaw
    u_dot = tau_u / m            # Acceleration from surge thrust
    r_dot = tau_r / I_z          # Angular acceleration from yaw torque

    xdot = ca.vertcat(x_dot, y_dot, psi_dot, u_dot, r_dot)

    # Assign model to acados structure
    model.x = states
    model.u = controls
    model.f_expl_expr = xdot
    model.f_impl_expr = states - xdot  # not used here but required

    return model

# Main routine to set up and solve the MPC
def MPC_solve(horizon = 20, 
        time = 2, 
        Q_weight = np.diag([10, 10, 10, 1, 1]),
        R_weight = np.diag([0.1, 0.1]),
        lower_bound_u = np.array([-5.0, -2.0]),
        upper_bound_u = np.array([5.0, 2.0]),
        input_constraints = np.array([0, 1])  # Which inputs are constrained):
        ):
    N_horizon = horizon   # Prediction steps
    T = time          # Total prediction time [s]
    dt = T / N_horizon

    # Load the model
    model = export_underwater_model()

    # Create OCP object
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N_horizon

    nx = model.x.size()[0]  # Number of states
    nu = model.u.size()[0]  # Number of control inputs

    # Define cost matrices
    Q = Q_weight     # Weight on states
    R = R_weight     # Weight on controls

    # Cost function setup using least-squares structure
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = np.block([
        [Q, np.zeros((nx, nu))],
        [np.zeros((nu, nx)), R]
    ])
    ocp.cost.W_e = Q

    # Cost function mapping matrices
    ocp.cost.Vx = np.hstack([np.eye(nx), np.zeros((nx, nu))])
    ocp.cost.Vu = np.hstack([np.zeros((nu, nx)), np.eye(nu)])
    ocp.cost.Vx_e = np.eye(nx)

    # Reference values for tracking (target state and control)
    x_ref = x_desired
    u_ref = u_desired
    ocp.cost.yref = np.concatenate((x_ref, u_ref))
    ocp.cost.yref_e = x_ref

    # Define a circular trajectory as reference #TODO input mpc_generated states
    # t_vals = np.linspace(0, T, N_horizon+1)
    # radius = 5.0
    # omega = 2 * np.pi / 10.0
    # x_refs = np.array([
    #     [radius * np.cos(omega * t),
    #      radius * np.sin(omega * t),
    #      omega * t,
    #      1.0,  # Constant surge velocity
    #      omega] for t in t_vals
    # ])
    u_refs = np.zeros((N_horizon, nu))  # assume zero desired thrust initially

    # Terminal reference
    ocp.cost.yref_e = x_refs[-1]

    # Input constraints (thruster and torque limits)
    ocp.constraints.lbu = lower_bound_u
    ocp.constraints.ubu = upper_bound_u
    ocp.constraints.idxbu = input_constraints

    # Initial state
    ocp.constraints.x0 = np.zeros(nx) 

    # Solver settings
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.tf = T

    # Create solver instance
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    # Set reference at each step
    for i in range(N_horizon):
        solver.set(i, 'yref', np.concatenate((x_ref, u_ref)))
    solver.set(N_horizon, 'yref', x_ref)

    # Solve the problem
    status = solver.solve()
    if status != 0:
        print(f"Solver failed with status {status}")

    # Extract solution (state and control trajectories)
    X = np.array([solver.get(i, 'x') for i in range(N_horizon + 1)])
    U = np.array([solver.get(i, 'u') for i in range(N_horizon)])

    return U

    # # Plot robot trajectory
    # plt.figure()
    # plt.plot(X[:,0], X[:,1], marker='o')
    # plt.title("AUV Trajectory")
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.axis('equal')
    # plt.grid()

    # # Plot control inputs
    # plt.figure()
    # plt.plot(U)
    # plt.title("Control Inputs (tau_u, tau_r)")
    # plt.xlabel("Prediction Step")
    # plt.legend(["tau_u", "tau_r"])
    # plt.grid()
    # plt.show()

# if __name__ == "__main__":
#     main()
