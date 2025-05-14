#!/usr/bin/env python3

from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy
import time
import numpy as np
from amarsmer_control import ROV
from hydrodynamic_model import hydrodynamic
from urdf_parser_py import urdf
from std_msgs.msg import String, Float32, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, Quaternion, Vector3
from scipy.spatial.transform import Rotation as R
from amarsmer_interfaces.srv import RequestPath

from ur_mpc import MPC_solve
from visualization_msgs.msg import Marker


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
        
        self.future = None # Usef for client requests

        self.timer = self.create_timer(0.1, self.move)

        ## Initiating variables

        # Pose
        self.desired_pose = None
        self.current_pose = None
        self.current_twist = None

        # Model
        self.mass = None
        self.added_masses = None
        self.viscous_drag = None
        self.quadratic_drag = None

        # MPC Parameters
        self.mpc_horizon = 20
        self.mpc_time = 2
        self.mpc_path = Path()



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

    def pose_callback(self, msg: PoseStamped):
        # Extract pose
        msg_pose = msg.pose

        self.desired_pose = self.pose_to_array(msg_pose)

    def odom_callback(self, msg: Odometry):
        # Extract pose
        msg_pose = msg.pose.pose

        self.current_pose = self.pose_to_array(msg_pose)

        # Extract twist
        twist = msg.twist.twist
        x = twist.linear.x
        y = twist.linear.y
        z = twist.linear.z

        p = twist.angular.x
        q = twist.angular.y
        r = twist.angular.z
        
        self.current_twist = [x,y,z,p,q,r]

        # self.get_logger().info(f"{self.pose}")
        # self.get_logger().info(f"{self.twist}")

    def create_pose_marker(self, pose):
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
        marker.pose = pose  # your geometry_msgs/Pose

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

        if self.mpc_path.poses: # Make sure the path is not empty

            self.create_pose_marker(self.mpc_path.poses[0].pose) # Display the current desired pose

            tau = MPC_solve(robot_mass = self.mass, 
                iz = self.inertia[-1], 
                horizon = self.mpc_horizon, 
                time = self.mpc_time, 
                Q_weight = np.diag([10, 10, 10, 1, 1]),
                R_weight = np.diag([0.1, 0.1]),
                lower_bound_u = np.array([-10.0, -2.0]),
                upper_bound_u = np.array([10.0, 2.0]),
                input_constraints = np.array([0, 1]),  # Which inputs are constrained
                path = self.mpc_path
                )

        cylinder_l = 0.6
        cylinder_r = 0.15

        B = np.array([[1,1],
                        [-cylinder_r,cylinder_r]])
        u = np.linalg.inv(B) @ tau

        # give thruster forces and joint angles
        self.rov.move([u[0],u[1],0,0],
                      [0 for i in range(1,5)])


rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()