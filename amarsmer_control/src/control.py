#!/usr/bin/env python3

from rclpy.node import Node
import rclpy
from amarsmer_control import ROV
from math import cos


class Controller(Node):
    def __init__(self):

        super().__init__('control', namespace='amarsmer')

        thrusters = [f'thruster{i}' for i in range(1,5)]
        joints = [f'thruster{i}_steering' for i in range(1,5)]
        self.rov = ROV(self, thrusters, joints)

        self.timer = self.create_timer(0.1, self.move)

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def move(self):
        if not self.rov.ready():
            return

        t = self.get_time()

        # give thruster forces and joint angles
        self.rov.move([10,10,10,10], [0.5*cos(i*t/10) for i in range(1,5)])



rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()
