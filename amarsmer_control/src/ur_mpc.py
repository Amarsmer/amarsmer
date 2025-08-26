# Remove syntax warnings from acados
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Regular imports
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
import math

# Utility to convert quaternion to yaw
def get_yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

# Model export
""" # Old version without hydrodynamic damping
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

    # Controls (can be expanded)
    tau_u = ca.SX.sym('tau_u')
    tau_r = ca.SX.sym('tau_r')
    controls = ca.vertcat(tau_u, tau_r)

    # Dynamics
    x_dot = u * ca.cos(psi)
    y_dot = u * ca.sin(psi)
    psi_dot = r
    u_dot = tau_u / robot_mass
    r_dot = tau_r / iz
    xdot = ca.vertcat(x_dot, y_dot, psi_dot, u_dot, r_dot)

    model.x = states
    model.u = controls
    model.f_expl_expr = xdot
    model.f_impl_expr = states - xdot

    return model
"""
def export_underwater_model(
    robot_mass=10.0,
    iz=5.0,
    # Added-mass matrix entries (positive values increase apparent inertia)
    a_u   = 2.0,    # added mass in surge
    a_r   = 0.5,    # added inertia in yaw

    # Linear damping (D). Use positive values; model subtracts D*nu.
    d_u   = 2.0,    # surge damping
    d_r   = 1.0,    # yaw damping
):

    """
    3-DOF kinematics (x,y,psi) + 2-DOF dynamics (u,r) with added mass.

    States: x, y, psi, u, r
    Inputs: tau_u, tau_r

    Equations:
      M * nu_dot = tau - D*nu - g(eta)
    where M = M_rb + M_a (2x2), nu = [u, r]^T.
    """

    model = AcadosModel()
    model.name = "ur_robot_model"

    # States
    x   = ca.SX.sym('x')
    y   = ca.SX.sym('y')
    psi = ca.SX.sym('psi')
    u   = ca.SX.sym('u')
    r   = ca.SX.sym('r')
    X   = ca.vertcat(x, y, psi, u, r)

    # State derivatives (for implicit form)
    x_dot_sym   = ca.SX.sym('x_dot')
    y_dot_sym   = ca.SX.sym('y_dot')
    psi_dot_sym = ca.SX.sym('psi_dot')
    u_dot_sym   = ca.SX.sym('u_dot')
    r_dot_sym   = ca.SX.sym('r_dot')
    Xdot = ca.vertcat(x_dot_sym, y_dot_sym, psi_dot_sym, u_dot_sym, r_dot_sym)

    # Controls
    tau_u = ca.SX.sym('tau_u')
    tau_r = ca.SX.sym('tau_r')
    U = ca.vertcat(tau_u, tau_r)

    # Kinematics (no sway)
    x_dot   = u * ca.cos(psi)
    y_dot   = u * ca.sin(psi)
    psi_dot = r

    # Build rigid-body mass and added-mass matrices (2x2 for [u, r])
    M_rb = ca.DM([[robot_mass, 0.0],
                  [0.0,        iz     ]])

    M_a = ca.DM([[a_u,  0],
                 [0, a_r ]])

    M = M_rb + M_a    # total mass matrix (2x2)

    # Damping matrix D (2x2)
    D = ca.DM([[d_u,  0],
               [0, d_r ]])

    # Velocity vector nu = [u, r]
    nu = ca.vertcat(u, r)

    # D*nu
    Dnu = ca.mtimes(D, nu)

    # Inputs (tau)
    tau = ca.vertcat(tau_u, tau_r)

    # Solve for nu_dot: M * nu_dot = tau - D*nu - g  =>  nu_dot = M^{-1} * (...)
    # Use casadi inverse (for 2x2 it's fine). If you prefer numerical stability
    # for larger matrices, use ca.solve(M, rhs) instead.
    nu_dot = ca.inv(M) @ (tau - Dnu)

    u_ddot = nu_dot[0]
    r_ddot = nu_dot[1]

    # assemble xdot
    xdot = ca.vertcat(x_dot, y_dot, psi_dot, u_ddot, r_ddot)

    # Pack model
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.f_expl_expr = xdot
    model.f_impl_expr = Xdot - xdot

    return model

class MPCController:
    def __init__(self, robot_mass=10, 
        iz=5, 
        a_u = 2.0,
        a_r = 0.5, 
        d_u = 2.0,
        d_r = 1.0, 
        horizon=20, 
        time=2.0,
        Q_weight=None, 
        R_weight=None, 
        input_bounds=None):

        self.mass = robot_mass
        self.iz = iz
        self.N = horizon
        self.T = time
        self.dt = time / horizon

        self.Q = Q_weight if Q_weight is not None else np.diag([10, 10, 10, 1, 1])
        self.R = R_weight if R_weight is not None else np.diag([0.1, 0.1])
        self.input_bounds = input_bounds if input_bounds is not None else {
            "lower": np.array([-5.0, -2.0]),
            "upper": np.array([5.0, 2.0]),
            "idx": np.array([0, 1])
        }

        self.model = export_underwater_model(self.mass, self.iz)
        self.ocp = self._build_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

    def _build_ocp(self):
        model = self.model
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu

        # Cost setup
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = np.eye(ny)
        ocp.cost.W[:nx, :nx] = self.Q
        ocp.cost.W[nx:, nx:] = self.R
        ocp.cost.W_e = self.Q
        ocp.constraints.x0 = np.zeros(5)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)

        ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
        ocp.cost.Vx_e = np.eye(nx)

        # Input constraints
        ocp.constraints.lbu = self.input_bounds["lower"]
        ocp.constraints.ubu = self.input_bounds["upper"]
        ocp.constraints.idxbu = self.input_bounds["idx"]

        # State constraints (enable x0 via lbx/ubx)
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx = -1e10 * np.ones(nx)
        ocp.constraints.ubx =  1e10 * np.ones(nx)

        # Solver setup
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.tf = self.T

        return ocp

    def update_weights(self, Q_weight=None, R_weight=None):
        if Q_weight is not None:
            self.Q = Q_weight
        if R_weight is not None:
            self.R = R_weight

        # Rebuild OCP and solver
        self.ocp = self._build_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

    def solve(self, path, x_current):
        poses = path.poses[:self.N + 1]
        if len(poses) < self.N + 1:
            poses += [poses[-1]] * (self.N + 1 - len(poses))

        x_refs, u_refs = [], []
        for i in range(self.N + 1):
            pose = poses[i].pose
            x = pose.position.x
            y = pose.position.y
            psi = get_yaw_from_quaternion(pose.orientation)
            psi = (psi + np.pi) % (2 * np.pi) - np.pi

            if i > 0:
                prev_pose = poses[i - 1].pose
                dx = x - prev_pose.position.x
                dy = y - prev_pose.position.y
                u = math.hypot(dx, dy) / self.dt
                psi_prev = get_yaw_from_quaternion(prev_pose.orientation)
                dpsi = (psi - psi_prev + np.pi) % (2 * np.pi) - np.pi
                r = dpsi / self.dt
            else:
                u, r = 0.0, 0.0

            x_refs.append([x, y, psi, u, r])
            if i < self.N:
                u_refs.append([0.0, 0.0])

        self.solver.set(0, 'lbx', x_current)
        self.solver.set(0, 'ubx', x_current)
        for i in range(self.N):
            yref = np.concatenate((x_refs[i], u_refs[i]))
            self.solver.set(i, 'yref', yref)
        self.solver.set(self.N, 'yref', np.array(x_refs[-1]))

        status = self.solver.solve()
        if status != 0:
            print(f"ACADOS solver failed with status {status}")

        U = np.array([self.solver.get(i, 'u') for i in range(self.N)])
        return U[0]