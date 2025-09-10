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

import numpy as np

def seabed_scanning(t):
    """
    Calculate non-differentiated reference positions at a single time t.
    
    Parameters
    ----------
    t : float
        Time at which to evaluate the trajectory.
    
    Returns
    -------
    xr, yr, zr, phir, thetar, psir : float
        Reference positions and orientations at time t.
    """
    velmin = 0.5
    r = velmin / (np.pi / 4)
    R = r
    rr = 1.0
    pas = 0.25

    # --- Positions ---
    if t <= 4:
        xr = velmin * t
        yr = 0.0
    elif t <= 6:
        xr = velmin*4 + r*np.sin((t-4)*np.pi/4)
        yr = r - r*np.cos((t-4)*np.pi/4)
    elif t <= 16:
        xr = r + velmin*4
        yr = r + velmin*(t-6)
    elif t <= 20:
        xr = velmin*4 + 2*r - r*np.cos((t-16)*np.pi/4)
        yr = r + velmin*10 + r*np.sin((t-16)*np.pi/4)
    elif t <= 30:
        xr = 3*r + velmin*4
        yr = r + velmin*10 - velmin*(t-20)
    elif t <= 40:
        xr = 3*r + velmin*4 + (velmin/np.sqrt(3))*(t-30)
        yr = r + (velmin/np.sqrt(3))*(t-30)
    elif t <= 40+12*np.pi:
        xr = 3*r + velmin*4 + 10*velmin/np.sqrt(3) + R*(1 + np.cos(np.pi + rr*velmin*(t-40)))
        yr = r + 10*velmin/np.sqrt(3) + R*np.sin(np.pi + rr*velmin*(t-40))
    else:
        # Default to last known point
        xr = 3*r + velmin*4 + 10*velmin/np.sqrt(3) - R
        yr = r + 10*velmin/np.sqrt(3)

    # --- Z position ---
    if t <= 30:
        zr = 1.0
    elif t <= 40:
        zr = 1.0 + (velmin/np.sqrt(3))*(t-30)
    elif t <= 40+4*np.pi:
        zr = 1.0 + 10*velmin/np.sqrt(3)
    elif t <= 40+12*np.pi:
        zr = 1.0 + 10*velmin/np.sqrt(3) - pas*velmin*(t-40-4*np.pi)
    else:
        zr = 1.0 + 10*velmin/np.sqrt(3) - pas*velmin*8*np.pi  # last known point

    # --- Rotations ---
    phir = 0.0  # Roll is zero in all phases

    if t <= 30:
        thetar = 0.0
    elif t <= 40:
        thetar = -np.arcsin(1/np.sqrt(3))
    elif t <= 40+4*np.pi:
        thetar = -np.pi/6
    elif t <= 40+12*np.pi:
        thetar = -np.pi/6 + (np.pi/3)*((t-40-4*np.pi)/(8*np.pi))
    else:
        thetar = np.pi/6  # last known

    if t <= 4:
        psir = 0.0
    elif t <= 6:
        psir = (t-4)*np.pi/4
    elif t <= 16:
        psir = np.pi/2
    elif t <= 20:
        psir = np.pi/2 - (t-16)*np.pi/4
    elif t <= 30:
        psir = -np.pi/2
    elif t <= 40:
        psir = np.pi/4
    elif t <= 40+12*np.pi:
        psir = rr*velmin*(t-40)
    else:
        psir = rr*velmin*12*np.pi 

    return xr, yr, zr, phir, thetar, psir
