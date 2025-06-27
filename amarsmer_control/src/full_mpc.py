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
"""
def export_underwater_model(robot_mass=10, 
                            inertia=np.ones(6), 
                            rg = np.zeros(3), 
                            rb = np.zeros(3),
                            added_masses_in = None, 
                            viscous_drag_in = None, 
                            quadratic_drag_in = None):

    model = AcadosModel()
    model.name = "6dof_robot_model"

    # Parameters, probably inefficient but makes the code more readable in the end
    I_xx = inertia[0]
    I_xy = inertia[1]
    I_xz = inertia[2]
    I_yy = inertia[3]
    I_yz = inertia[4]
    I_zz = inertia[5]

    xg = rg[0]
    yg = rg[1]
    zg = rg[2]

    xb = rb[0]
    yb = rb[1]
    zb = rb[2]

    # added_masses = ca.vertcat(*[x for x in added_masses_in])
    # viscous_drag = ca.vertcat(*[x for x in viscous_drag_in])
    # quadratic_drag = ca.vertcat(*[x for x in quadratic_drag_in])

    added_masses = ca.MX(added_masses_in)
    viscous_drag = ca.MX(viscous_drag_in)
    quadratic_drag = ca.MX(quadratic_drag_in)

    # States
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    z = ca.MX.sym('z')
    phi = ca.MX.sym('phi')
    theta = ca.MX.sym('theta')
    psi = ca.MX.sym('psi')

    u = ca.MX.sym('u')
    v = ca.MX.sym('v')
    w = ca.MX.sym('w')
    p = ca.MX.sym('p')
    q = ca.MX.sym('q')
    r = ca.MX.sym('r')

    # nu2 = ca.vertcat(p, q, r)

    eta = ca.vertcat(x, y, z, phi, theta, psi)
    nu = ca.vertcat(u, v, w, p, q, r)

    nu2 = nu[3:]

    states = ca.vertcat(eta, nu)

    # Controls
    X = ca.MX.sym('X')
    Y = ca.MX.sym('Y')
    Z = ca.MX.sym('Z')
    K = ca.MX.sym('K')
    M = ca.MX.sym('M')
    N = ca.MX.sym('N')

    controls = ca.vertcat(X, Y, Z, K, M ,N)

    # Math tools
    Rz = ca.vertcat(
        ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
        ca.horzcat(ca.sin(psi),  ca.cos(psi), 0),
        ca.horzcat(0,            0,           1))

    Ry = ca.vertcat(
        ca.horzcat(ca.cos(theta),  0, ca.sin(theta)),
        ca.horzcat(0,              1,             0),
        ca.horzcat(-ca.sin(theta), 0, ca.cos(theta)))

    Rx = ca.vertcat(
        ca.horzcat(1,           0,            0),
        ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
        ca.horzcat(0, ca.sin(phi),  ca.cos(phi)))

    J_1 = Rz @ Ry @ Rx 

    J_2 = ca.vertcat(
        ca.horzcat(1, ca.sin(pĥi)*ca.tan(theta),   ca.cos(phi)*ca.tan(theta)),
        ca.horzcat(0,               ca.cos(phi),                -ca.sin(phi)),
        ca.horzcat(0, ca.sin(phi)/ca.cos(theta),  ca.cos(phi)/ca.cos(theta)))

    J = ca.vertcat(
        ca.horzcat(J_1, ca.MX.zeros(3, 3)),
        ca.horzcat(ca.MX.zeros(3, 3), J_2))

    # Dynamics
    eta_dot = J @ nu

    Ib = ca.vertcat(
        ca.horzcat(I_xx, -I_xy, -I_xz),
        ca.horzcat(-I_xy, I_yy, -I_yz),
        ca.horzcat(-I_xz, -I_yz, I_zz))

    M_rb_11 = m*ca.MX.eye(3) 
    M_rb_12 = -m*ca.skew(rg)
    M_rb_21 = m*ca.skew(rg)
    M_rb_22 = Ib

    M_rb = ca.vertcat(
        ca.horzcat(M_rb_11, M_rb_12),
        ca.horzcat(M_rb_21, M_rb_22))

    M_a = -ca.diag(added_masses)

    C_rb_11 = m*ca.skew(nu2)
    C_rb_12 = -m*ca.skew(nu2) @ ca.skew(rg)
    C_rb_21 = m*ca.skew(rg) @ ca.skew(nu2)
    C_rb_22 = ca.skew(nu2) @ Ib

    C_rb = ca.vertcat(
        ca.horzcat(C_rb_11, C_rb_12),
        ca.horzcat(C_rb_21, C_rb_22))

    nu_dot = ca.vertcat(u_dot, v_dot, w_dot, p_dot, q_dot, r_dot)
    xdot = ca.vertcat(eta_dot, nu_dot)

    model.x = states
    model.u = controls
    model.f_expl_expr = xdot
    model.f_impl_expr = states - xdot

    return model
    """
class MPCController:
    def __init__(self, robot_mass=10, 
                 inertia=np.zeros(6), 
                 rg = np.zeros(3), 
                 rb = np.zeros(3), 
                 added_masses = None, 
                 viscous_drag = None, 
                 quadratic_drag = None, 
                 horizon=20, 
                 time=2.0,
                 Q_weight=None, 
                 R_weight=None, 
                 input_bounds=None):

        self.mass = robot_mass
        self.inertia = inertia
        self.rg = rg
        self.rb = rb
        self.added_masses = added_masses
        self.viscous_drag = viscous_drag
        self.quadratic_drag = quadratic_drag

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

        self.model = self.export_underwater_model()
        self.ocp = self._build_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

    def export_underwater_model(self):

        model = AcadosModel()
        model.name = "6dof_robot_model"

        ### Parameters, probably inefficient but makes the code more readable in the end
        m = ca.MX(self.mass)

        I_xx = ca.MX(self.inertia[0])
        I_xy = ca.MX(self.inertia[1])
        I_xz = ca.MX(self.inertia[2])
        I_yy = ca.MX(self.inertia[3])
        I_yz = ca.MX(self.inertia[4])
        I_zz = ca.MX(self.inertia[5])

        rg = ca.MX(self.rg)
        xg = rg[0]
        yg = rg[1]
        zg = rg[2]

        rb = ca.MX(self.rb)
        xb = rb[0]
        yb = rb[1]
        zb = rb[2]

        # added_masses = ca.vertcat(*[x for x in added_masses_in])
        # viscous_drag = ca.vertcat(*[x for x in viscous_drag_in])
        # quadratic_drag = ca.vertcat(*[x for x in quadratic_drag_in])

        added_masses = ca.MX(self.added_masses)
        viscous_drag = ca.MX(self.viscous_drag)
        quadratic_drag = ca.MX(self.quadratic_drag)

        ### States
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        z = ca.MX.sym('z')
        phi = ca.MX.sym('phi')
        theta = ca.MX.sym('theta')
        psi = ca.MX.sym('psi')

        u = ca.MX.sym('u')
        v = ca.MX.sym('v')
        w = ca.MX.sym('w')
        p = ca.MX.sym('p')
        q = ca.MX.sym('q')
        r = ca.MX.sym('r')

        # nu2 = ca.vertcat(p, q, r)

        eta = ca.vertcat(x, y, z, phi, theta, psi)
        nu = ca.vertcat(u, v, w, p, q, r)

        nu_1 = nu[:3]
        nu_2 = nu[3:]

        states = ca.vertcat(eta, nu)

        ### Controls
        X = ca.MX.sym('X')
        Y = ca.MX.sym('Y')
        Z = ca.MX.sym('Z')
        K = ca.MX.sym('K')
        M = ca.MX.sym('M')
        N = ca.MX.sym('N')

        tau = ca.vertcat(X, Y, Z, K, M ,N)

        ### Math tools
        Rz = ca.vertcat(
            ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
            ca.horzcat(ca.sin(psi),  ca.cos(psi), 0),
            ca.horzcat(0,            0,           1))

        Ry = ca.vertcat(
            ca.horzcat(ca.cos(theta),  0, ca.sin(theta)),
            ca.horzcat(0,              1,             0),
            ca.horzcat(-ca.sin(theta), 0, ca.cos(theta)))

        Rx = ca.vertcat(
            ca.horzcat(1,           0,            0),
            ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
            ca.horzcat(0, ca.sin(phi),  ca.cos(phi)))

        J_1 = Rz @ Ry @ Rx 

        J_2 = ca.vertcat(
            ca.horzcat(1, ca.sin(phi)*ca.tan(theta),   ca.cos(phi)*ca.tan(theta)),
            ca.horzcat(0,               ca.cos(phi),                -ca.sin(phi)),
            ca.horzcat(0, ca.sin(phi)/ca.cos(theta),  ca.cos(phi)/ca.cos(theta)))

        J = ca.vertcat(
            ca.horzcat(J_1, ca.MX.zeros(3, 3)),
            ca.horzcat(ca.MX.zeros(3, 3), J_2))

        eta_dot = J @ nu 

        ### Dynamics
        Ib = ca.vertcat(
            ca.horzcat(I_xx, -I_xy, -I_xz),
            ca.horzcat(-I_xy, I_yy, -I_yz),
            ca.horzcat(-I_xz, -I_yz, I_zz))

        ## M matrix

        # Rigid body
        M_rb_11 = m*ca.MX.eye(3) 
        M_rb_12 = -m*ca.skew(rg)
        M_rb_21 = m*ca.skew(rg)
        M_rb_22 = Ib

        M_rb = ca.vertcat(
            ca.horzcat(M_rb_11, M_rb_12),
            ca.horzcat(M_rb_21, M_rb_22))

        # Added masses
        M_a = -ca.diag(added_masses)

        # Complete
        M = M_rb + M_a

        ## C matrix

        # Rigid body
        C_rb_11 = m*ca.skew(nu_2)
        C_rb_12 = -m*ca.skew(nu_2) @ ca.skew(rg)
        C_rb_21 = m*ca.skew(rg) @ ca.skew(nu_2)
        C_rb_22 = ca.skew(nu_2) @ Ib

        C_rb = ca.vertcat(
            ca.horzcat(C_rb_11, C_rb_12),
            ca.horzcat(C_rb_21, C_rb_22))

        # Added masses
        lin_added_masses = added_masses[:3]
        ang_added_masses = added_masses[3:]

        C_a_11 = ca.MX.zeros(3,3)
        C_a_12 = ca.skew(lin_added_masses * nu_1) 
        C_a_21 = C_a_12 
        C_a_22 = ca.skew(ang_added_masses * nu_2) 

        C_a = ca.vertcat(
            ca.horzcat(C_a_11, C_a_12),
            ca.horzcat(C_a_21, C_a_22))

        # Complete
        C = C_rb + C_a

        ## D matrix

        D = -ca.diag(viscous_drag) - ca.diag(quadratic_drag * ca.fabs(nu))

        ## G vector

        W = m * 9.81 # Weight - m * g
        B = 1026 * 9.81 * 60 * 3.14 # Buoyancy - rho * g * (robot's volume approximation)

        G = ca.vertcat((W - B) * ca.sin(theta),
                -(W - B) * ca.cos(theta) * ca.sin(phi),
                -(W - B) * ca.cos(theta) * ca.cos(phi),
                -(yg * W - zb * B) * ca.cos(theta) * ca.cos(phi) + (zg * W - zb * B) * ca.cos(theta) * ca.sin(phi),
                (zg * W - zb * B) * ca.sin(theta) + (xg * W - xb * B) * ca.cos(theta) * ca.cos(phi),
                -(xg * W - xb * B) * ca.cos(theta) * ca.sin(phi) - (yg * W - yb * B) * ca.sin(theta))

        ### State-space
        nu_dot = ca.inv(M) @ (tau - (C @ nu + D @ nu + G))
        x_dot = ca.vertcat(eta_dot, nu_dot)

        model.x = states
        model.u = tau
        model.f_expl_expr = x_dot
        model.f_impl_expr = states - x_dot

        return model

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