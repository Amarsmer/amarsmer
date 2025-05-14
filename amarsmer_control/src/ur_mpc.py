#!/usr/bin/env python3

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt

# Convert quaternion to yaw angle (psi) manually (no external libraries needed)
def get_yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

# Define the nonlinear underwater robot model
def export_underwater_model(robot_mass=10, iz=5):
    model = AcadosModel()
    model.name = "ur_robot_model"

    # States
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    psi = ca.SX.sym('psi')
    u = ca.SX.sym('u')
    r = ca.SX.sym('r')
    states = ca.vertcat(x, y, psi, u, r)

    # Controls
    tau_u = ca.SX.sym('tau_u')
    tau_r = ca.SX.sym('tau_r')
    controls = ca.vertcat(tau_u, tau_r)

    # Dynamics
    m = robot_mass
    I_z = iz

    x_dot = u * ca.cos(psi)
    y_dot = u * ca.sin(psi)
    psi_dot = r
    u_dot = tau_u / m
    r_dot = tau_r / I_z
    xdot = ca.vertcat(x_dot, y_dot, psi_dot, u_dot, r_dot)

    model.x = states
    model.u = controls
    model.f_expl_expr = xdot
    model.f_impl_expr = states - xdot

    return model

# Main MPC solver setup and run
def MPC_solve(robot_mass=10, 
              iz=5, 
              horizon=20, 
              time=2,
              Q_weight=np.diag([10, 10, 10, 1, 1]),
              R_weight=np.diag([0.1, 0.1]),
              lower_bound_u=np.array([-5.0, -2.0]),
              upper_bound_u=np.array([5.0, 2.0]),
              input_constraints=np.array([0, 1]),
              path=None):

    N_horizon = horizon
    T = time
    dt = T / N_horizon

    model = export_underwater_model(robot_mass, iz)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N_horizon

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    Q = Q_weight
    R = R_weight

    # Define yref mapping
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = np.eye(ny)
    ocp.cost.W[:nx, :nx] = Q
    ocp.cost.W[nx:, nx:] = R
    ocp.cost.W_e = Q

    # Correct Vx and Vu dimension setup
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])  # (ny, nx)
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])  # (ny, nu)
    ocp.cost.Vx_e = np.eye(nx)

    # Reference trajectory setup
    x_refs = []
    u_refs = []

    poses = path.poses[:N_horizon+1]
    if len(poses) < N_horizon + 1:
        poses += [poses[-1]] * (N_horizon + 1 - len(poses))

    for i in range(N_horizon + 1):
        pose = poses[i].pose
        x = pose.position.x
        y = pose.position.y
        psi = get_yaw_from_quaternion(pose.orientation)

        if i > 0:
            prev_pose = poses[i - 1].pose
            dx = x - prev_pose.position.x
            dy = y - prev_pose.position.y
            u = math.hypot(dx, dy) / dt
            psi_prev = get_yaw_from_quaternion(prev_pose.orientation)
            r = (psi - psi_prev) / dt
        else:
            u = 0.0
            r = 0.0

        x_refs.append([x, y, psi, u, r])
        if i < N_horizon:
            u_refs.append([0.0, 0.0])

    # Initial state and terminal cost
    ocp.constraints.x0 = np.array(x_refs[0])
    ocp.cost.yref = np.zeros(ny)  # default
    ocp.cost.yref_e = np.array(x_refs[-1])

    ocp.constraints.lbu = lower_bound_u
    ocp.constraints.ubu = upper_bound_u
    ocp.constraints.idxbu = input_constraints

    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.tf = T

    solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    for i in range(N_horizon):
        yref_i = np.concatenate((x_refs[i], u_refs[i]))
        solver.set(i, 'yref', yref_i)
    solver.set(N_horizon, 'yref', np.array(x_refs[-1]))

    status = solver.solve()
    if status != 0:
        print(f"Solver failed with status {status}")

    X = np.array([solver.get(i, 'x') for i in range(N_horizon + 1)])
    U = np.array([solver.get(i, 'u') for i in range(N_horizon)])

    return U[0]
