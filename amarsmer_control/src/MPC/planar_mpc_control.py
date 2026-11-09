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
        self.mpcData_publisher = self.create_publisher(Float32MultiArray, "/amarsmer/ctrlData", 10)
        self.thruster_input_publisher = self.create_publisher(Float32MultiArray, "/thruster_input", 10)

        self.timer = self.create_timer(self.dt, self.move)

        # MPC Parameters
        self.mpc_horizon = 20
        self.mpc_time = 2.5
        self.mpc_path = Path()
        linear_bound = 40.0
        self.input_bounds = {"lower": np.array([-linear_bound]*self.nb_thrusters),
                             "upper": np.array([linear_bound]*self.nb_thrusters),
                             "idx":   np.arange(self.nb_thrusters)
                             }
                             
        self.Q_weight = np.diag([70, # x
                                 70, # y 
                                 25, # cos psi
                                 25, # sin psi
                                 1, # u
                                 1, # v
                                 1  # r
                                 ])
        
        self.R_weight = np.diag([0.008]*self.nb_thrusters) # equal for every thruster

        # Initialize MPC solver
        self.controller = None #Updated at the start of spin
        self.state = None

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def ctrl_callback(self, msg):
        # self.state = np.array(msg.state.data)
        x,y,psi,u,v,r = msg.state.data
        self.state = np.array([x,y, np.cos(psi), np.sin(psi), u, v, r])

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
        u = np.zeros(self.nb_thrusters).reshape(-1,1)

        if self.mpc_path.poses: # Make sure the path is not empty
            u = self.controller.solve(path=self.mpc_path, x_current=self.state)

        # Check for solver error
        status = self.controller.error_msg
        if status != "":
            self.get_logger().info(f'Status error: {status}\nDefaulting to previous output')

        # Compute loss function
        target_temp = cf.compute_target(self.mpc_path, self.dt, sc = True)
        target = np.array(target_temp).reshape(-1,1)
        error = self.state.reshape(-1,1) - target
        
        crit_x = error.transpose() @ self.Q_weight @ error
        crit_u = u.transpose() @ self.R_weight @ u
        loss = [crit_x, crit_u]

        # Publish controller data
        publisher_msg = Float32MultiArray()
        publisher_msg.data = loss
        self.mpcData_publisher.publish(publisher_msg)

        # Publish thruster input
        publisher_msg = Float32MultiArray()
        publisher_msg.data = u
        self.thruster_input_publisher.publish(publisher_msg)

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()