#!/usr/bin/env python3

# rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy

# Common python libraries
import numpy as np
import time

# ROS2 msg libraries
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose

# Custom libraries
import planar_mpc
from amarsmer_interfaces.msg import InCtrl
import custom_functions as cf

class Controller(Node):
    def __init__(self):

        super().__init__('mpc_control', namespace='amarsmer')

        self.declare_parameter('nb_thrusters', 2) 
        self.nb_thrusters = self.get_parameter('nb_thrusters').get_parameter_value().integer_value

        self.declare_parameter('dt', 0.05)
        self.dt = self.get_parameter('dt').get_parameter_value().double_value

        self.InCtrl_subscriber = self.create_subscription(InCtrl, '/amarsmer/InCtrl', self.ctrl_callback, 10)
        self.thruster_input_publisher = self.create_publisher(Float32MultiArray, "/thruster_input", 10)

        self.timer = self.create_timer(self.dt, self.move)

        # MPC Parameters
        self.mpc_horizon = 10
        self.mpc_time = 2.0
        self.mpc_path = Path()
        linear_bound = 40.0
        self.input_bounds = {"lower": np.array([-linear_bound]*self.nb_thrusters),
                             "upper": np.array([linear_bound]*self.nb_thrusters),
                             "idx":   np.arange(self.nb_thrusters)
                             }
                             
        self.Q_weight = np.diag([50, # x
                                 50, # y 
                                 40, # psi
                                 1, # u
                                 1, # v
                                 1  # r
                                 ])
        
        self.R_weight = np.diag([0.015]*self.nb_thrusters) # equal for every thruster

        # Initialize MPC solver
        self.controller = None #Updated at the start of spin
        self.state = None

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def ctrl_callback(self, msg):
        self.state = np.array(msg.state.data)
        self.mpc_path = msg.path

    def move(self):
        # Read YAML file for robot's properties
        mass, inertia, added_masses, viscous_drag, _ = cf.read_model()

        if self.controller is None:
            self.controller = planar_mpc.MPCController(robot_mass = mass,
                                                iz = inertia[-1],
                                                a_u = added_masses[0],
                                                a_v = added_masses[1],
                                                a_r = added_masses[5],
                                                d_u = viscous_drag[0],
                                                d_v = viscous_drag[1],
                                                d_r = viscous_drag[5],
                                                horizon = self.mpc_horizon,
                                                time = self.mpc_time,
                                                Q_weight = self.Q_weight,
                                                R_weight = self.R_weight,
                                                input_bounds = self.input_bounds,
                                                thrusters = self.nb_thrusters
                                                )

        if self.state is None:
            return

        # MPC control
        u = np.zeros(self.nb_thrusters)

        if self.mpc_path.poses: # Make sure the path is not empty
            u = self.controller.solve(path=self.mpc_path, x_current=self.state)

        # self.get_logger().info(f'\nState: {self.state}\n\nPath*: {self.mpc_path.poses[0]} \n\nU: {u}')
        status = self.controller.error_msg
        if status != "":
            self.get_logger().info(f'Status error: {status}\nDefaulting to previous output')

        publisher_msg = Float32MultiArray()
        publisher_msg.data = u
        self.thruster_input_publisher.publish(publisher_msg)

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()