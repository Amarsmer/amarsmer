#!/usr/bin/env python3

# rclpy
import rclpy
from rclpy.node import Node

# Common python libraries
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime

# ROS2 msg libraries
from std_msgs.msg import String, Float32, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker

# Custom libraries
from urdf_parser_py import urdf
import full_mpc as mpc
from amarsmer_control import ROV
from amarsmer_interfaces.srv import RequestPath
import custom_functions as cf

class Controller(Node):
    def __init__(self):

        super().__init__('mpc_control', namespace='amarsmer')

        ######### Robot #########

        self.rov = ROV(self, thrust_visual = True)

        ######### ROS Interactions #########

        # Publisher and subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/amarsmer/odom', self.odom_callback, 10)
        self.pose_arrow_publisher = self.create_publisher(Marker, "/pose_arrow", 10)

        # Create a client for path request
        self.client = self.create_client(RequestPath, '/path_request')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")
        
        self.future = None # Used for client requests

        # self.timer = self.create_timer(0.01, self.move)

        ######### Initiating variables #########

        # Pose
        self.current_pose = None
        self.current_twist = None

        # MPC Parameters
        self.mpc_horizon = 5
        self.mpc_time = 2
        self.mpc_path = Path()
        thrust = 40.0
        torque = 15.0
        self.input_bounds = {"lower": np.array([-thrust, -thrust, -thrust, -torque, -torque, -torque]), # minimal force and torque
                             "upper": np.array([thrust, thrust, thrust, torque, torque, torque]), # maximal force and torque
                             "idx":   np.array([0, 1, 2, 3, 4, 5]) # index of the constrained inputs
                             }

        # Cost matrices
        self.Q_weight = np.diag([50, # x
                                 50, # y
                                 50, # z
                                 50, # phi
                                 50, # theta
                                 50, # psi
                                 20, # u
                                 20, # v
                                 20, # w
                                 20, # p
                                 20, # q
                                 20]) # r

        self.R_weight = np.diag([0.05, # X
                                 0.05, # Y
                                 0.05, # Z
                                 0.1, # K
                                 0.1, # M
                                 0.1]) # N


        # Initialize MPC solver
        self.controller = None
        self.ready = False

        # while self.ready == False:
        #     self.initialize_MPC()

        self.timer = self.create_timer(0.01, self.move)

        # Initialize monitoring values
        self.monitoring = []
        self.monitoring.append(['x','y','psi','x_d','y_d','psi_d','u1','u2','t'])

        self.date = datetime.today().strftime('%Y_%m_%d-%H_%M_%S')

    def __del__(self):
        self.get_logger().info("Saving monitoring data")
        title = self.date +'-mpc_data'
        np.save(title, self.monitoring)

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def odom_callback(self, msg: Odometry):
        # Extract pose
        msg_pose = msg.pose.pose

        self.current_pose = mpc.pose_to_array(msg_pose)

        # Extract twist
        twist = msg.twist.twist
        u = twist.linear.x
        v = twist.linear.y
        w = twist.linear.z

        p = twist.angular.x
        q = twist.angular.y
        r = twist.angular.z
        
        self.current_twist = [u,v,w,p,q,r]

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

    # def initialize_MPC(self):
    #     # self.get_logger().info(f"Parsed: {self.rov.parsed()}")
    #     # self.get_logger().info(f"Ready: {self.rov.ready()}")

    #     if not self.rov.parsed():
    #         return

    #     if not self.rov.ready():
    #         return

        
    #     self.controller = mpc.MPCController(robot_mass = self.rov.mass,
    #                                     inertia = self.rov.inertia,
    #                                     rg = self.rov.rg,
    #                                     rb = self.rov.rb,
    #                                     added_masses = self.rov.added_masses,
    #                                     viscous_drag = self.rov.viscous_drag,
    #                                     quadratic_drag = self.rov.quadratic_drag,
    #                                     horizon = self.mpc_horizon, 
    #                                     time = self.mpc_time, 
    #                                     Q_weight = self.Q_weight,
    #                                     R_weight = self.R_weight,
    #                                     input_bounds = self.input_bounds
    #                                     )

    #     self.ready = True

    def move(self):

        if not self.rov.parsed() or not self.rov.ready():
            return

        if self.controller == None: # Initialize controller
            self.controller = mpc.MPCController(robot_mass = self.rov.mass,
                                            inertia = self.rov.inertia,
                                            rg = self.rov.rg,
                                            rb = self.rov.rb,
                                            added_masses = self.rov.added_masses,
                                            viscous_drag = self.rov.viscous_drag,
                                            quadratic_drag = self.rov.quadratic_drag,
                                            horizon = self.mpc_horizon, 
                                            time = self.mpc_time, 
                                            Q_weight = self.Q_weight,
                                            R_weight = self.R_weight,
                                            input_bounds = self.input_bounds
                                            )

            self.t0 = self.get_time()

        ######### Path request #########

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
            
        t = self.get_time() - self.t0

        # Send new request
        request = RequestPath.Request()
        request.path_request.data = np.linspace(t, t + self.mpc_time, int(self.mpc_horizon) + 1, dtype=float)

        self.future = self.client.call_async(request)

        ######### MPC control #########

        # Initialize mpc output
        init_tau = False

        # Update pose for mpc input
        if self.current_pose is not None and self.current_twist is not None:
            x_current = np.array(self.current_pose # eta
                                  + self.current_twist) # nu

        # Solve MPC
        
        if self.mpc_path.poses: # Make sure the path is not empty
            desired_pose = self.mpc_path.poses[0].pose
            self.create_pose_marker(desired_pose) # Display the current desired pose
            tau = self.controller.solve(path=self.mpc_path, x_current=x_current)
            init_tau = True
        
        # self.get_logger().info(f"{self.mpc_path.poses}")

        ######### Robot Control #########

        # Define thrust allocation matrix and use it to apply tau on thrusters
        B = self.rov.B

        # self.get_logger().info(f"B \n {B}")
        # self.get_logger().info(f"pinvB \n {np.linalg.pinv(B)}")
        # self.get_logger().info(f"tau \n {tau}")

        if not init_tau:
            return
        
        u = np.linalg.pinv(B) @ tau

        # give thruster forces and joint angles
        self.rov.move([u[0], u[1], u[2], u[3]],
                      [0 for i in range(1,5)])

        """
        if self.mpc_path.poses:
            self.get_logger().info(f"[DEBUG] Path length: {len(self.mpc_path.poses)}")
            self.get_logger().info(f"[DEBUG] x_current: {x_current}")
            for i, p in enumerate(self.mpc_path.poses[:3]):  # Just first 3
                self.get_logger().info(f"[DEBUG] Pose[{i}] = {p.pose.position.x}, {p.pose.position.y}, {p.pose.position.z}")
        """

        """
        # TODO: Implement the new monitoring code
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
        """
        

if __name__ == '__main__':

    rclpy.init()
    node = Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
