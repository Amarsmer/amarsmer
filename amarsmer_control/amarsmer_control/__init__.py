#!/usr/bin/env python3

from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState


class ROV:

    def __init__(self, node: Node,
                 thrusters = [], joints = []):

        self.joints = joints
        self.q = [0 for _ in self.joints]

        self.pose = None
        self.twist = None

        self.odom_sub = node.create_subscription(Odometry, 'odom', self.odom_cb, 1)

        self.thruster_pub = []
        for thr in thrusters:
            self.thruster_pub.append(node.create_publisher(Float64, 'cmd_'+thr, 1))

        self.joint_pub = []
        for joint in joints:
            self.joint_pub.append(node.create_publisher(Float64, 'cmd_'+joint, 1))

    def ready(self):
        return self.pose is not None and self.twist is not None

    def odom_cb(self, odom: Odometry):
        self.pose = odom.pose.pose
        self.twist = odom.twist.twist

    def joint_cb(self, joints: JointState):

        for i,thruster in enumerate(self.joints):
            if thruster in joints.name:
                idx = joints.name.index(thruster)
                self.q[i] = joints.position[idx]

    def move(self, forces, angles):

        msg = Float64()
        for i, val in enumerate(forces):
            msg.data = float(val)
            self.thruster_pub[i].publish(msg)
        for i, val in enumerate(angles):
            msg.data = float(val)
            self.joint_pub[i].publish(msg)
