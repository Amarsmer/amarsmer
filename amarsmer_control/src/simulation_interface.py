#!/usr/bin/env python3

# rclpy
from rclpy.node import Node
import rclpy

# Common python libraries
import numpy as np

# ROS2 msg libraries
from std_msgs.msg import Bool, Float32MultiArray

# Custom libraries
from amarsmer_control import ROV
import custom_functions as cf

class Controller(Node):
    def __init__(self):

        super().__init__('sim_interface', namespace='amarsmer')

        self.declare_parameter('nb_thrusters', 2) 
        self.nb_thrusters = self.get_parameter('nb_thrusters').get_parameter_value().integer_value

        self.declare_parameter('dt', 0.05)
        self.dt = self.get_parameter('dt').get_parameter_value().double_value

        self.declare_parameter('spawn_pose', '0.0 0.0 0.0 0.0 0.0 0.0')
        spwn = np.array([float(x)for x in self.get_parameter('spawn_pose').value.split()])
        cf.set_pose_gz(spwn)

        self.rov = ROV(self, thrust_visual = True)

        self.thruster_input_sub = self.create_subscription(Float32MultiArray, "/thruster_input", self.thr_input_callback,10)

        self.ready_publisher = self.create_publisher(Bool, '/amarsmer/controller_ready', 10)

        self.timer = self.create_timer(self.dt, self.move)
        self.thr_input = [0]*self.nb_thrusters
        self.thrust_timer = None
        self.sent_ready = False

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def thr_input_callback(self, msg: Float32MultiArray):
        self.thr_input = msg.data
        self.thrust_timer = self.get_time()

    def move(self):
        if not self.rov.ready():
            return

        if not self.sent_ready:
            msg = Bool()
            msg.data = True
            self.ready_publisher.publish(msg)
            self.get_logger().info(f'Ready publishing: {msg.data}')
            self.sent_ready = True

        # Safety: stop the robot if no input has been received in some time
        if self.thrust_timer is not None:
            input_silence = self.get_time() - self.thrust_timer
            if input_silence > 1.0:
                self.thr_input = [0]*self.nb_thrusters

        # Apply force to thrusters
        self.rov.move([*self.thr_input],
                      [0]*self.nb_thrusters)

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()