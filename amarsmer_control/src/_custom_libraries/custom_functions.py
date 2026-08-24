#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import numpy as np
import sympy as sp
import os
from ament_index_python.packages import get_package_share_directory
import yaml

#################### Yaml reading ####################

def read_model():
    # Locate YAML file
    pkg_path = get_package_share_directory("amarsmer_control")

    yaml_path = os.path.join(
        pkg_path,
        "config",
        "robot.yaml"
    )

    ## Load YAML
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    ## Robot properties
    robot = cfg["robot"]

    mass = robot["mass"]
    z_cog = robot["z_cog"]

    # Inertia tensor coefficients
    inertia = np.array(list(
    robot["inertia"].values()
    ))

    ## Hydrodynamics
    hydro = cfg["hydrodynamics"]

    # Added mass coefficients
    added_mass = np.array(list(
        hydro["added_mass"].values()
    ))

    # Linear damping coefficients
    linear_damping = np.array(list(
        hydro["linear_damping"].values()
    ))

    # Quadratic damping coefficients
    quadratic_damping = np.array(list(
        hydro["quadratic_damping"].values()
    ))

    return mass, inertia, added_mass, linear_damping, quadratic_damping

#################### Gradient computation ####################

def build_grad(B, mass, added_mass, inertia, dampening, radius, Qw, Rw):
    
    ########## Variable init ########
    
    cos = sp.cos
    sin = sp.sin
    
    #### Time
    t = sp.Symbol('t')

    ####State space
    # Position
    x = sp.Function('x')(t)
    y = sp.Function('y')(t)
    psi = sp.Function('psi')(t)

    # position derivatives
    x_dot,y_dot,psi_dot = sp.symbols('x_dot y_dot psi_dot')

    # Speeds
    u = sp.Function('u')(t)
    v = sp.Function('v')(t)
    r = sp.Function('r')(t)

    # accelerations
    u_dot,v_dot,r_dot = sp.symbols('u_dot v_dot r_dot')

    #### Dynamic Model
    
    m = sp.Symbol('m')

    # M matrix
    Xu_dot, Yv_dot, Nr_dot, Iz = sp.symbols('Xu_dot Yv_dot Nr_dot Iz')

    # D matrix
    Xu, Yv, Nr = sp.symbols('Xu Yv Nr')
    
    # Control inputs
    R = sp.Symbol('R')
    
    nb_thr = B.shape[1]
    
    U = sp.Matrix(
        sp.symbols(f'U1:{nb_thr+1}')
    )
    
    ########## Model ##########
    
    # Kinematic
    eta = sp.Matrix([[x],
                     [y],
                     [psi]])
    
    nu = sp.Matrix([[u],
                    [v],
                    [r]])
    
    J = sp.Matrix([[cos(psi), -sin(psi), 0],
                   [sin(psi),  cos(psi), 0],
                   [   0    ,     0    , 1]])

    eta_dot = J*nu
    
    x_dot, y_dot, psi_dot = eta_dot
    
    # Dynamic
    M = sp.Matrix([[m-Xu_dot, 0, 0],
                   [0, m-Yv_dot, 0],
                   [0, 0, Iz-Nr_dot]])
    
    C = sp.Matrix([[0, -m*r, Yv_dot*v],
                   [m*r, 0, -Xu_dot*u],
                   [-Yv_dot*v, Xu_dot*u, 0]])
    
    D = -sp.Matrix([[Xu, 0, 0],
                   [0, Yv, 0],
                   [0, 0, Nr]])
    
    Tau = B*U

    nu_dot = M.inv() * (Tau - C*nu - D*nu)
    
    u_dot, v_dot, r_dot = nu_dot
    
    # State space
    X = sp.Matrix([[x],
                   [y],
                   [psi],
                   [u],
                   [v],
                   [r]])
    
    X_dot = sp.Matrix([[x_dot],
                       [y_dot],
                       [psi_dot],
                       [u_dot],
                       [v_dot],
                       [r_dot]])
    
    # Differentiate
    X_ddot = sp.diff(X_dot, t)
    X_ddot = X_ddot.subs({
        sp.diff(x,t) : x_dot,
        sp.diff(y,t) : y_dot,
        sp.diff(psi,t) : psi_dot,
        sp.diff(u,t) : u_dot,
        sp.diff(v,t) : v_dot,
        sp.diff(r,t) : r_dot
        })
    
    ########## Gradient computation ##########
    
    # Target
    # x_d,y_d,psi_d, u_d, v_d, r_d = sp.symbols('x_d y_d psi_d u_d v_d r_d')
    # T = sp.Matrix([[x_d],
    #                [y_d],
    #                [psi_d],
    #                [u_d],
    #                [v_d],
    #                [r_d]])

    e_x, e_y, e_psi, e_u, e_v, e_r = sp.symbols('e_x e_y e_psi e_u e_v e_r')
    E = sp.Matrix([[e_x],
                   [e_y],
                   [e_psi],
                   [e_u],
                   [e_v],
                   [e_r]])

    # First order gradient
    grad_U = sp.Matrix.hstack(*(sp.diff(X_dot, u) for u in U))

    # Second order gradient
    gradd_U = sp.Matrix.hstack(*(sp.diff(X_ddot, u) for u in U))
    
    # Create gradient function
    dt, alpha1, alpha2 = sp.symbols('dt alpha1 alpha2')
    
    gradxJ = 2 * (Qw * E)
    graduJ = 2 * (Rw * U)
        
    time_gradient = alpha1 * dt * grad_U + alpha2 * 0.5*dt**2 * gradd_U
    
    full_gradient = time_gradient.T *gradxJ + graduJ
    
    # Copy full symbolic expression for supervision purposes
    symb_Ud = grad_U.copy()
    symb_Udd = gradd_U.copy()
    symb_full = full_gradient.copy()
    
    # Substitute parameters with given values
    added_mass_symbols = (Xu_dot, Yv_dot, Nr_dot)
    damping_symbols = (Xu, Yv, Nr)
    
    subs = {}
    subs.update({m: mass})
    subs.update(zip(added_mass_symbols, added_mass))
    subs.update({Iz: inertia})
    subs.update(zip(damping_symbols, dampening))
    subs.update({R: radius})
    
    f_num = full_gradient.subs(subs)
    f = sp.lambdify([X,E,U,dt,alpha1,alpha2], f_num, modules='numpy')
    
    return f, symb_full, symb_Ud, symb_Udd
    
#################### Gazebo interaction ####################
def pause_gz(pause):
    os.system(f'gz service -s /world/ocean/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --req \'pause: {pause}\' --timeout 1000')

def set_pose_gz(pose):
    x,y,z,phi,theta,psi = pose.astype(float)
    qx, qy, qz, qw = R.from_euler('xyz',[phi, theta, psi]).as_quat()
    
    pause_gz(True)
    os.system(f'gz service -s /world/ocean/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --req \'name: "amarsmer" position: {{ x: {x}, y: {y}, z: {z} }} orientation: {{ x: {qx}, y: {qy}, z: {qz}, w: {qw} }}\' --timeout 1000')
    pause_gz(False)

def set_current_gz(x, y, z):
    os.system(f'gz topic -t "/ocean_current" -m gz.msgs.Vector3d -p "x: {x}, y: {y}, z: {z}"')

#################### ROS2 interaction ####################
def odometry(msg, quat = False):
    # Extract pose
    msg_pose = msg.pose.pose

    # Extract position
    x = msg_pose.position.x
    y = msg_pose.position.y
    z = msg_pose.position.z

    # Extract orientation (quaternion)
    qx = msg_pose.orientation.x
    qy = msg_pose.orientation.y
    qz = msg_pose.orientation.z
    qw = msg_pose.orientation.w

    if quat: # Use quaternion directly
        pose = [x,y,z,qx,qy,qz,qw]

    else:
        # Convert quaternion to roll, pitch, yaw
        rot = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

        pose = [x,y,z,roll,pitch,yaw]

    # Extract twist
    twist = msg.twist.twist

    u = twist.linear.x
    v = twist.linear.y
    w = twist.linear.z

    p = twist.angular.x
    q = twist.angular.y
    r = twist.angular.z

    twist = [u,v,w,p,q,r]

    return pose, twist

def planeFromQuaternion(inState):
    # Extract position
    x,y,z = inState[0:3]

    #Convert and extract euler angles
    phi,theta,psi = R.from_quat(inState[3:7]).as_euler('xyz')

    # Extract robot frame's speeds
    u,v,r = inState[7:]

    return np.array([x,y,psi,u,v,r]).reshape(-1,1)

def quaternion_multiply(q0, q1): # From https://docs.ros.org/en/foxy/Tutorials/Intermediate/Tf2/Quaternion-Fundamentals.html
    """
    Multiplies two quaternions.

    Input
    :param q0: A 4 element array containing the first quaternion (q01, q11, q21, q31)
    :param q1: A 4 element array containing the second quaternion (q02, q12, q22, q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    """
    # Extract the values from q0
    w0 = q0[0]
    x0 = q0[1]
    y0 = q0[2]
    z0 = q0[3]

    # Extract the values from q1
    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]

    # Computer the product of the two quaternions, term by term
    q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion

def quaternion_error(q2, q1): # Returns "q2-q1"
    q1[3] *= -1 # Negate for inverse

    return quaternion_multiply(q2, q1)

def make_pose(pose_list, quat = False): #TODO: Make it usable in 3D
    pose = Pose()

    if quat:
        # Position
        pose.position.x = float(pose_list[0])
        pose.position.y = float(pose_list[1])
        pose.position.z = float(pose_list[2])

        pose.orientation.x = float(pose_list[3])
        pose.orientation.y = float(pose_list[4])
        pose.orientation.z = float(pose_list[5])
        pose.orientation.w = float(pose_list[6])

    else:
        x = pose_list[0]
        y = pose_list[1]
        theta = pose_list[2]

        # Position
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0

        # Orientation from yaw (theta)
        qz = np.sin(theta / 2.0)
        qw = np.cos(theta / 2.0)

        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = qz
        pose.orientation.w = qw

    return pose

def create_pose_marker(inPose, inPub):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.5  # shaft length
    marker.scale.y = 0.05  # shaft diameter
    marker.scale.z = 0.05  # head diameter
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.pose = inPose

    marker.id = 0
    marker.lifetime.sec = 0  # persistent

    inPub.publish(marker)

def compute_target(path, dt):

        def get_yaw_from_quaternion(q):
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return np.arctan2(siny_cosp, cosy_cosp)

        # Extract current target and next step target
        poses = path.poses[:2]
        present = poses[0].pose
        future = poses[1].pose

        # Compute position
        x = future.position.x
        y = future.position.y
        psi = get_yaw_from_quaternion(future.orientation)
        psi = (psi + np.pi) % (2 * np.pi) - np.pi

        # Compute speeds
        dx = x - present.position.x
        dy = y - present.position.y
        u = np.hypot(dx, dy) / dt

        psi_prev = get_yaw_from_quaternion(present.orientation)

        psi_mid = (psi + psi_prev) / 2.0
        dx_b =  np.cos(psi_mid) * dx + np.sin(psi_mid) * dy
        dy_b = -np.sin(psi_mid) * dx + np.cos(psi_mid) * dy
        v = dy_b / dt 

        dpsi = (psi - psi_prev + np.pi) % (2 * np.pi) - np.pi
        r = dpsi / dt
        
        return [x, y, psi, u, v, r]


#################### Trajectory generation ####################
def seabed_scanning(t):
    """
    Calculate non-differentiated reference positions at a single time t.
    
    Parameters
    ----------
    t : float
        Time at which to evaluate the trajectory.
    
    Returns
    -------
    xr, yr, zr, phir, thetar, psir : float
        Reference positions and orientations at time t.
    """
    velmin = 0.5
    r = velmin / (np.pi / 4)
    R = r
    rr = 1.0
    pas = 0.25

    # --- Positions ---
    if t <= 4:
        xr = velmin * t
        yr = 0.0
    elif t <= 6:
        xr = velmin*4 + r*np.sin((t-4)*np.pi/4)
        yr = r - r*np.cos((t-4)*np.pi/4)
    elif t <= 16:
        xr = r + velmin*4
        yr = r + velmin*(t-6)
    elif t <= 20:
        xr = velmin*4 + 2*r - r*np.cos((t-16)*np.pi/4)
        yr = r + velmin*10 + r*np.sin((t-16)*np.pi/4)
    elif t <= 30:
        xr = 3*r + velmin*4
        yr = r + velmin*10 - velmin*(t-20)
    elif t <= 40:
        xr = 3*r + velmin*4 + (velmin/np.sqrt(3))*(t-30)
        yr = r + (velmin/np.sqrt(3))*(t-30)
    elif t <= 40+12*np.pi:
        xr = 3*r + velmin*4 + 10*velmin/np.sqrt(3) + R*(1 + np.cos(np.pi + rr*velmin*(t-40)))
        yr = r + 10*velmin/np.sqrt(3) + R*np.sin(np.pi + rr*velmin*(t-40))
    else:
        # Default to last known point
        xr = 3*r + velmin*4 + 10*velmin/np.sqrt(3) - R
        yr = r + 10*velmin/np.sqrt(3)

    # --- Z position ---
    if t <= 30:
        zr = 1.0
    elif t <= 40:
        zr = 1.0 + (velmin/np.sqrt(3))*(t-30)
    elif t <= 40+4*np.pi:
        zr = 1.0 + 10*velmin/np.sqrt(3)
    elif t <= 40+12*np.pi:
        zr = 1.0 + 10*velmin/np.sqrt(3) - pas*velmin*(t-40-4*np.pi)
    else:
        zr = 1.0 + 10*velmin/np.sqrt(3) - pas*velmin*8*np.pi  # last known point

    # --- Rotations ---
    phir = 0.0  # Roll is zero in all phases

    if t <= 30:
        thetar = 0.0
    elif t <= 40:
        thetar = -np.arcsin(1/np.sqrt(3))
    elif t <= 40+4*np.pi:
        thetar = -np.pi/6
    elif t <= 40+12*np.pi:
        thetar = -np.pi/6 + (np.pi/3)*((t-40-4*np.pi)/(8*np.pi))
    else:
        thetar = np.pi/6  # last known

    if t <= 4:
        psir = 0.0
    elif t <= 6:
        psir = (t-4)*np.pi/4
    elif t <= 16:
        psir = np.pi/2
    elif t <= 20:
        psir = np.pi/2 - (t-16)*np.pi/4
    elif t <= 30:
        psir = -np.pi/2
    elif t <= 40:
        psir = np.pi/4
    elif t <= 40+12*np.pi:
        psir = rr*velmin*(t-40)
    else:
        psir = rr*velmin*12*np.pi 

    return xr, yr, zr, phir, thetar, psir
