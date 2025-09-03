#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R

def odometry(msg):
    # Extract pose
    msg_pose = msg.pose.pose

    # Extract position
    x = msg_pose.position.x
    y = msg_pose.position.y
    z = msg_pose.position.z

    # Extract orientation (quaternion)
    qx = msg_pose.orientation.x
    qy = msg_pose.orientation.y
    qz = msg_pose.orientation.z
    qw = msg_pose.orientation.w

    # Convert quaternion to roll, pitch, yaw
    rot = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

    pose = [x,y,z,roll,pitch,yaw]

    # Extract twist
    twist = msg.twist.twist
    u = twist.linear.x
    v = twist.linear.y
    w = twist.linear.z

    p = twist.angular.x
    q = twist.angular.y
    r = twist.angular.z

    twist = [u,v,w,p,q,r]

    return pose, twist


def make_pose(pose_list):
    x = pose_list[0]
    y = pose_list[1]
    theta = pose_list[2]

    pose = Pose()

    # Position
    pose.position.x = x
    pose.position.y = y
    pose.position.z = 0.0

    # Orientation from yaw (theta)
    qz = math.sin(theta / 2.0)
    qw = math.cos(theta / 2.0)

    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = qz
    pose.orientation.w = qw

    return pose

def create_pose_marker(inPose, inPub):
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

    inPub.publish(marker)

    # return marker