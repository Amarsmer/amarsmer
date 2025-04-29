#!/usr/bin/env python3

from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
from urdf_parser_py import urdf
from numpy import array


def convert(v):
    return array([v.x,v.y,v.z])


class ROV:

    def __init__(self, node: Node,
                 thrusters = [], joints = [],
                 model = None,
                 thrust_visual = False):

        self.display_wrench = thrust_visual # Creating and storing the bool so it may be accessed and changed during simulation if the need arises

        self.joints = joints
        self.q = [0 for _ in self.joints]

        self.p = None
        self.R = None
        self.v = None
        self.w = None

        self.odom_sub = node.create_subscription(Odometry, 'odom', self.odom_cb, 1)
        self.js_sub = node.create_subscription(JointState, 'joint_states', self.joint_cb, 1)

        self.thruster_pub = []
        for thr in thrusters:
            self.thruster_pub.append(node.create_publisher(Float64, 'cmd_'+thr, 1))

        self.joint_pub = []
        for joint in joints:
            self.joint_pub.append(node.create_publisher(Float64, 'cmd_'+joint, 1))

        # Wrench publishers, used to display thrust
        self.wrench_pub = []
        for thr in thrusters: # A bit redundant with thruster_pub but easier to read and correct
            self.wrench_pub.append(node.create_publisher(WrenchStamped, 'amarsmer_' + thr + '_wrench', 1)) 

    def ready(self):
        return self.p is not None

    def odom_cb(self, odom: Odometry):

        self.p = convert(odom.pose.pose.position)
        q = odom.pose.pose.orientation
        self.R = Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()

        self.v = convert(odom.twist.twist.linear)
        self.w = convert(odom.twist.twist.angular)

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

        # Redundant again but future me might thank me
        if self.display_wrench:
            wrench_msg = WrenchStamped()
            for i, val in enumerate(forces):
                wrench_msg.header.frame_id = "amarsmer/thruster"+str(i+1)
                # Create and publish WrenchStamped
                wrench_msg.wrench.force.x = float(-val)
                wrench_msg.wrench.force.y = 0.0
                wrench_msg.wrench.force.z = 0.0
                wrench_msg.wrench.torque.x = 0.0
                wrench_msg.wrench.torque.y = 0.0
                wrench_msg.wrench.torque.z = 0.0
                self.wrench_pub[i].publish(wrench_msg)
