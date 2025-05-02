#!/usr/bin/env python3

from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy
from amarsmer_control import ROV
from math import cos
from urdf_parser_py import urdf
from std_msgs.msg import String
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped


class Controller(Node):
    def __init__(self):

        super().__init__('mpc_control', namespace='amarsmer')

        thrusters = [f'thruster{i}' for i in range(1,5)]
        joints = [f'thruster{i}_steering' for i in range(1,5)]
        self.rov = ROV(self, thrusters, joints, thrust_visual = True)

        self.robot = None
        self.robot_sub = self.create_subscription(String, 'robot_description',
                                                  self.read_model,
                                                  QoSProfile(depth=1,
                                                  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        self.timer = self.create_timer(0.1, self.move)

        self.time_publisher = self.create_publisher(Float32, '/request', 10)
        self.pose_subscriber = self.create_subscription(PoseStamped, '/desired_pose', self.pose_callback, 10)

        self.desired_pose = PoseStamped()


    def read_model(self, msg):

        self.robot = urdf.Robot.from_xml_string(msg.data)

        print(len(self.robot.joints), 'joints')
        print(len(self.robot.links), 'links')


        for j in self.robot.joints:
            if j.type == 'continuous':
                print('found thruster', j.name)

        l1 = self.robot.links[0]
        print('mass:',l1.inertial.mass)

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
        print(Ma)
        print(Dl)
        print(Dq)

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def pose_callback(self, msg: PoseStamped):
        self.desired_pose = msg
        print(self.desired_pose)

    def move(self):
        if not self.rov.ready() or self.robot is None:
            return

        t = self.get_time()
        self.time_publisher.publish(Float32(data=t))

        #TODO MPC control

        # give thruster forces and joint angles
        self.rov.move([10,-10,0,0],
                      [t for i in range(1,5)])


rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()
