# Remove syntax warnings from acados
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Regular imports
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def pose_to_array(msg_pose): # Used to convert pose msg to a regular array
    # Extract position
    x = msg_pose.position.x
    y = msg_pose.position.y
    z = msg_pose.position.z

    # Extract orientation (quaternion)
    qx = msg_pose.orientation.x
    qy = msg_pose.orientation.y
    qz = msg_pose.orientation.z
    qw = msg_pose.orientation.w

    # Convert quaternion to roll, pitch, yaw
    rot = R.from_quat([qx, qy, qz, qw])
    phi, theta, psi = rot.as_euler('xyz', degrees=False)

    return [x,y,z,phi,theta,psi]

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

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

        self.available_solver = False

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

        self.Q = Q_weight
        self.R = R_weight
        self.input_bounds = input_bounds

        self.model = self.export_underwater_model()
        self.ocp = self._build_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        self.available_solver = True

    def export_underwater_model(self):

        model = AcadosModel()
        model.name = "robot_model"

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
        ocp.constraints.x0 = np.zeros(12)
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

    def solve(self, path, x_current):
        poses = path.poses[:self.N + 1]
        if len(poses) < self.N + 1:
            poses += [poses[-1]] * (self.N + 1 - len(poses))

        x_refs, u_refs = [], []
        for i in range(self.N + 1):
            pose = poses[i].pose

            x,y,z,phi,theta,psi = pose_to_array(pose)
            
            if i > 0:
                prev_pose = poses[i - 1].pose

                prev_x, prev_y, prev_z, prev_phi, prev_theta, prev_psi = pose_to_array(prev_pose)

                # Linear velocities
                dx = x - prev_x
                dy = y - prev_y
                dz = z - prev_z
                
                u = dx / self.dt
                v = dy / self.dt
                w = dz / self.dt

                # Angular velocities
                dphi = wrap_angle(phi - prev_phi)
                dtheta = wrap_angle(theta - prev_theta)
                dpsi = wrap_angle(psi - prev_psi)
                
                p = dphi / self.dt
                q = dtheta / self.dt
                r = dpsi / self.dt

            else:
                u,v,w,p,q,r = np.zeros(6)

            x_refs.append([x, y, z, phi, theta, psi, u, v, w, p, q, r])
            if i < self.N:
                u_refs.append(np.zeros(6))

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