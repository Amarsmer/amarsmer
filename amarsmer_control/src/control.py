#!/usr/bin/env python3

from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy
from amarsmer_control import ROV
from math import cos
from urdf_parser_py import urdf
from std_msgs.msg import String


class Controller(Node):
    def __init__(self):

        super().__init__('control', namespace='amarsmer')

        thrusters = [f'thruster{i}' for i in range(1,5)]
        joints = [f'thruster{i}_steering' for i in range(1,5)]
        self.rov = ROV(self, thrusters, joints)

        self.robot = None
        self.robot_sub = self.create_subscription(String, 'robot_description',
                                                  self.read_model,
                                                  QoSProfile(depth=1,
                                                    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL))

        self.timer = self.create_timer(0.1, self.move)


    def read_model(self, msg):

        self.robot = urdf.Robot.from_xml_string(msg.data)

        print(len(self.robot.joints), 'joints')
        print(len(self.robot.links), 'links')

        j1 = self.robot.joints[0]
        print(j1.parent, j1.child, j1.axis)

        l1 = self.robot.links[0]
        print('mass:',l1.inertial.mass)




    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def move(self):
        if not self.rov.ready() or self.robot is None:
            return

        t = self.get_time()

        # give thruster forces and joint angles
        self.rov.move([10,10,10,10],
                      [0.5*cos(i*t/10) for i in range(1,5)])


rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()
