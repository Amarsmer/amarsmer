#!/usr/bin/env python3

# rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy

# Common python libraries
import numpy as np

# ROS2 msg libraries
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose

# Custom libraries
import PID
from amarsmer_interfaces.msg import InCtrl
import custom_functions as cf

class Controller(Node):
    def __init__(self):

        super().__init__('control', namespace='amarsmer')
        self.declare_parameter('nb_thrusters', 2) 
        self.nb_thrusters = self.get_parameter('nb_thrusters').get_parameter_value().integer_value

        self.declare_parameter('dt', 0.05)
        self.dt = self.get_parameter('dt').get_parameter_value().double_value

        self.InCtrl_subscriber = self.create_subscription(InCtrl, '/amarsmer/InCtrl', self.ctrl_callback, 10)
        self.thruster_input_publisher = self.create_publisher(Float32MultiArray, "/thruster_input", 10)

        self.timer = self.create_timer(self.dt, self.move)

        # Parameters
        self.mpc_time = 2.0
        self.PID_path = Path()
        linear_bound = 40.0

        self.thruster_limits = {"min": np.array([-linear_bound]*self.nb_thrusters),   
                                "max": np.array([linear_bound]*self.nb_thrusters)}

        radius = 0.15
        length = 0.3

        if self.nb_thrusters == 2:

            self.outer_gains = {'x': (3., 0.01, 0.),
                                'psi': (1.2, 0.01, 0.)}

            self.inner_gains = {'u': (1., 0., 0.),
                                'r': (1., 0., 0.)}

            self.B_matrix = B = np.array([[1.        ,1.],
                                        [0.        ,0.],
                                        [radius,-radius]])

        else:
            self.get_logger().error('uvr architecture currently unsupported')
            return
        
        # Initialize PID solver
        self.controller = None #Updated at the start of spin
        self.state = None

    def ctrl_callback(self, msg):
        self.state = np.array(msg.state.data)
        self.PID_path = msg.path

    def move(self):
        if self.controller is None:
            self.controller = PID.PIDLoS(dt = self.dt,
                                         B = self.B_matrix,
                                         outer_gains = self.outer_gains,
                                         inner_gains = self.inner_gains,
                                         thruster_limits = self.thruster_limits
                                         )

        if self.state is None:
            return

        # PID control
        u = [0]*self.nb_thrusters

        if self.PID_path.poses: # Make sure the path is not empty
            target = cf.compute_target(self.controller_path, self.dt)
            u,_ = self.controller.compute(current_state, target[:3])

        # Publish computed thrust
        publisher_msg = Float32MultiArray()
        publisher_msg.data = u
        self.thruster_input_publisher.publish(publisher_msg)

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()