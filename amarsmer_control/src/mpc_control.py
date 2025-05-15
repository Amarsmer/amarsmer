#!/usr/bin/env python3

# rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy

# Common python libraries
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# ROS2 msg libraries
from std_msgs.msg import String, Float32, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker

# Custom libraries
from urdf_parser_py import urdf
from hydrodynamic_model import hydrodynamic
from ur_mpc import MPC_solve
from amarsmer_control import ROV
from amarsmer_interfaces.srv import RequestPath

class Controller(Node):
    def __init__(self):

        super().__init__('mpc_control', namespace='amarsmer')

        thrusters = [f'thruster{i}' for i in range(1,5)]
        joints = [f'thruster{i}_steering' for i in range(1,5)]
        self.rov = ROV(self, thrusters, joints, thrust_visual = True)

        self.robot = None

        # Publisher and subscribers
        self.robot_sub = self.create_subscription(String, 'robot_description',
                                                  self.read_model,
                                                  QoSProfile(depth=1,
                                                  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        self.odom_subscriber = self.create_subscription(Odometry, '/amarsmer/odom', self.odom_callback, 10)
        self.pose_arrow_publisher = self.create_publisher(Marker, "/pose_arrow", 10)

        # Create a client for path request
        self.client = self.create_client(RequestPath, '/path_request')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")
        
        self.future = None # Used for client requests

        self.timer = self.create_timer(0.1, self.move)

        ## Initiating variables

        # Pose
        self.current_pose = None
        self.current_twist = None

        # Model
        self.mass = None
        self.cylinder_l = None
        self.cylinder_r = None
        self.added_masses = None
        self.viscous_drag = None
        self.quadratic_drag = None

        # MPC Parameters
        self.mpc_horizon = 40
        self.mpc_time = 3
        self.mpc_path = Path()

        # Initialize monitoring values
        self.monitoring = []
        self.monitoring.append(['x','y','psi','x_d','y_d','psi_d','u1','u2','t'])

    def read_model(self, msg):

        self.robot = urdf.Robot.from_xml_string(msg.data)

        print(len(self.robot.joints), 'joints')
        print(len(self.robot.links), 'links')


        for j in self.robot.joints:
            if j.type == 'continuous':
                print('found thruster', j.name)

        l1 = self.robot.links[0]
        # print('mass:',l1.inertial.mass)
        # print('rg:', l1.inertial.origin)
        # print('inertia:', l1.inertial.inertia)

        # Read the robot's dynamic parameters
        Ma = [0]*6
        Dl = [0]*6
        Dq = [0]*6

        for gz in self.robot.gazebos:
            for plugin in gz.findall('plugin'):
                if 'Hydrodynamics' in plugin.get('name'):
                    for i,(axis,force) in enumerate(('xU','yV','zW', 'kP', 'mQ', 'nR')):
                        for tag in plugin.findall(axis+force):
                            Dl[i] = float(tag.text)
                        for tag in plugin.findall(f'{axis}{force}abs{force}'):
                            Dq[i] = float(tag.text)
                        for tag in plugin.findall(f'{axis}Dot{force}'):
                            Ma[i] = float(tag.text)

        # print(Ma)
        # print(Dl)
        # print(Dq)

        # Update robot's model for hydrodynamic computation
        self.mass = l1.inertial.mass
        self.added_masses = Ma
        self.viscous_drag = Dl
        self.quadratic_drag = Dq

        read_inertia = l1.inertial.inertia
        self.inertia = [
            read_inertia.ixx,
            read_inertia.ixy,
            read_inertia.ixz,
            read_inertia.iyy,
            read_inertia.iyz,
            read_inertia.izz
        ]

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def pose_to_array(self, msg_pose): # Used to convert pose msg to a regular array
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
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

        return [x,y,z,roll,pitch,yaw]

    def odom_callback(self, msg: Odometry):
        # Extract pose
        msg_pose = msg.pose.pose

        self.current_pose = self.pose_to_array(msg_pose)

        # Extract twist
        twist = msg.twist.twist
        u = twist.linear.x
        v = twist.linear.y
        w = twist.linear.z

        p = twist.angular.x
        q = twist.angular.y
        r = twist.angular.z
        
        self.current_twist = [u,v,w,p,q,r]

        # self.get_logger().info(f"{self.pose}")
        # self.get_logger().info(f"{self.twist}")

    def create_pose_marker(self, inPose):
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

        self.pose_arrow_publisher.publish(marker)

    def move(self):
        if not self.rov.ready() or self.robot is None:
            return

        t = self.get_time()

        # Check if previous future is still pending
        if self.future is not None:
            if self.future.done():
                try:
                    result = self.future.result()
                    if result is not None:
                        self.mpc_path = result.path
                        # self.get_logger().info(f"Received path with {len(self.mpc_path.poses)} poses.")
                    else:
                        self.get_logger().error("Service returned None.")
                except Exception as e:
                    self.get_logger().error(f"Service call raised exception: {e}")
                finally:
                    self.future = None
                return

        # Send new request
        request = RequestPath.Request()
        request.path_request.data = np.linspace(t, t + self.mpc_time, int(self.mpc_horizon) + 1, dtype=float)

        self.future = self.client.call_async(request)
        
        # tau = hydrodynamic(rg = np.zeros(3), 
        #          rb = np.zeros(3), 
        #          eta = self.current_pose, 
        #          nu = np.zeros(6), 
        #          nudot = np.zeros(6), 
        #          added_masses = self.added_masses, 
        #          viscous_drag = self.viscous_drag, 
        #          quadratic_drag = self.quadratic_drag, 
        #          inertia=self.inertia)

        # self.get_logger().info(f"{tau}")

        # MPC control
        tau = np.zeros(2)

        if self.current_pose is not None and self.current_twist is not None:
            x_current = np.array([self.current_pose[0], # x
                                  self.current_pose[1], # y
                                  self.current_pose[5], # yaw
                                  self.current_twist[0], # u
                                  self.current_twist[5]]) # r

        if self.mpc_path.poses: # Make sure the path is not empty

            desired_pose = self.mpc_path.poses[0].pose
            self.create_pose_marker(desired_pose) # Display the current desired pose

            tau = MPC_solve(robot_mass = self.mass, 
                iz = self.inertia[-1], 
                horizon = self.mpc_horizon, 
                time = self.mpc_time, 
                Q_weight = np.diag([10, 10, 10, 5, 20]),
                R_weight = np.diag([0.1, 1.5]),
                lower_bound_u = np.array([-40.0, -15.0]),
                upper_bound_u = np.array([40.0, 15.0]),
                input_constraints = np.array([0, 1]),  # Index of which inputs are constrained
                path = self.mpc_path,
                x_current = x_current
                )

        cylinder_l = 0.6
        cylinder_r = 0.15

        # Define thrust allocation matrix and use it to apply tau on thrusters
        B = np.array([[1        ,1],
                     [cylinder_r,-cylinder_r]]) # Note that the current frame is NOT NED so the y and z axis are reversed
        u = np.linalg.inv(B) @ tau

        # give thruster forces and joint angles
        self.rov.move([u[0],u[1],0,0],
                      [0 for i in range(1,5)])

        # Update and save monitoring metrics to be graphed later
        if self.mpc_path.poses:
            x_m = self.current_pose[0]
            y_m = self.current_pose[1]
            psi_m = self.current_pose[5]

            x_d_m = desired_pose.position.x
            y_d_m = desired_pose.position.y

            q = desired_pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            psi_d_m = math.atan2(siny_cosp, cosy_cosp)

            self.monitoring.append([x_m, y_m, psi_m, x_d_m, y_d_m , psi_d_m, u[0],u[1], t])
            np.save('mpc_data', self.monitoring)


rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()