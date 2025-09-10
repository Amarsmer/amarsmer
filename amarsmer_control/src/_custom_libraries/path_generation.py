#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32, Float32MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R
from amarsmer_interfaces.srv import RequestPath
import math
import functions as f

"""
Creates a services that handle path generation requests. Receives a an array of time values and responds with the associated path.
"""

class PathGeneration(Node):
    def __init__(self):
        super().__init__('path_generation')

        # Declare parameters
        self.declare_parameter('display_log', False)
        self.display_log = self.get_parameter('display_log').value

        # Service
        self.path_service = self.create_service(RequestPath, '/path_request', self.generate_path)

    def single_pose(self, t: float, path_shape = 'sin') -> PoseStamped:
        """
        Generate a helical pose for a given time t.
        """
        radius = 4.0        # meters
        depth_per_circle = 2.0  # meters
        num_turns = 3
        total_length = 2 * np.pi * num_turns

        # Normalize t
        # t = (t / self.total_time) * total_length
        if path_shape == 'circle':
            # Circle
            t *= 0.08
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            # z = -(depth_per_circle * t) / (2 * np.pi)
            z = 0.0

            dx = -radius * np.sin(t)
            dy = radius * np.cos(t)
            yaw = np.arctan2(dy, dx)
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi # Normalize

        if path_shape == 'straight_line':
            # Straight line
            x = 0.4*t
            y = 2.0
            z = 0.0
            yaw = 0
        
        if path_shape == 'sin':
            # Sin line
            a = 3
            f = 0.1
            vx = 0.3

            x = vx*t
            y = 0.0 + a * np.sin(f*t)
            z = 0.0

            dx = vx
            dy = a * f * np.cos(f*t)
            yaw = np.arctan2(dy, dx)

        if path_shape == 'square':
            # Square wave
            period = 0.01
            amplitude = 2.0
            heading_dt = 0.01
            t /= 2
            def get_xy(s):
                x = s
                cycles = math.floor(s / 3)
                y = 2.0 if cycles % 2 == 0 else -2.0
                return x, y

            # Current position
            x, y = get_xy(t)
            x = float(x)
            y = float(y)
            z = 0.0

            # Compute heading using forward difference
            x_fwd, y_fwd = get_xy(t + heading_dt)
            dx = x_fwd - x
            dy = y_fwd - y
            yaw = math.atan2(dy, dx)

        if path_shape == 'kin_square':
            # Kinematic square wave

            segment_length = 6.0
            surge_speed = 0.4
            z = 0.0
            t *= 1.5
            # Time per segment
            segment_time = segment_length / surge_speed

            # Determine which segment we're in
            segment_index = int(t // segment_time)
            t_in_segment = t % segment_time
            
            directions = [
                (1, 0),     # +X
                (0, 1),     # +Y
                (1, 0),    # +X
                (0, -1),    # -Y
            ]
            yaws = [0, math.pi/2, 0, -math.pi/2]

            # Get direction and yaw
            dir_idx = segment_index % 4
            dx, dy = directions[dir_idx]
            yaw = yaws[dir_idx]

            # Total completed segments
            completed = segment_index

            # Compute cumulative position
            x, y = 0.0, 0.0
            for i in range(completed):
                dxi, dyi = directions[i % 4]
                x += dxi * segment_length
                y += dyi * segment_length

            # Move along current segment
            x += dx * surge_speed * t_in_segment
            y += dy * surge_speed * t_in_segment

        if path_shape == 'seabed_scanning':
            x,y,z,roll,pitch,yaw = f.seabed_scanning(t)
            x = float(x)
            y = float(y)
            z = 0.0
            yaw = float(yaw)
            

        # Create and return pose

        quat = R.from_euler('zyx', [yaw, 0.0, 0.0]).as_quat()

        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        return pose

    def generate_path(self, request, response):
        if self.display_log:
            self.get_logger().info(f"Received path_request of type: {type(request.path_request)}")

        path_msg = Path()
        path_msg.header.frame_id = 'world'

        for t in request.path_request.data:
            temp_pose = self.single_pose(t)
            temp_pose.header.stamp = self.get_clock().now().to_msg()
            path_msg.poses.append(temp_pose)

        response.path = path_msg

        if self.display_log:
            self.get_logger().info("Returning response...")

        return response

    def single_request(self, msg: Float32):
        time_request = msg.data
        desired_pose = self.single_pose(time_request)
        desired_pose.header.stamp = self.get_clock().now().to_msg()
        self.pose_publisher.publish(desired_pose)

def main(args=None):
    rclpy.init(args=args)
    node = PathGeneration()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
