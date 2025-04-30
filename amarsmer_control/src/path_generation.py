#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32
import numpy as np
from scipy.spatial.transform import Rotation as R

class PathGeneration(Node):
    def __init__(self):
        super().__init__('path_generation')

        # Declare parameters
        self.declare_parameter('total_time', 30.0)
        self.declare_parameter('dt', 0.1)

        self.total_time = self.get_parameter('total_time').value
        self.dt = self.get_parameter('dt').value

        # Publishers
        self.path_publisher = self.create_publisher(Path, 'full_path', 10)
        self.pose_publisher = self.create_publisher(PoseStamped, 'desired_pose', 10)

        # Subscriber
        self.create_subscription(Float32, '/request', self.request_callback, 10)

        # Generate and publish the full path once
        self.generate_and_publish_path()

    def path(self, t: float) -> PoseStamped:
        """
        Generate a helical pose for a given time t.
        """
        radius = 4.0        # meters
        depth_per_circle = 2.0  # meters
        num_turns = 3
        total_length = 2 * np.pi * num_turns

        # Normalize t
        t_normalized = (t / self.total_time) * total_length

        x = radius * np.cos(t_normalized)
        y = radius * np.sin(t_normalized)
        z = -(depth_per_circle * t_normalized) / (2 * np.pi)

        dx = -radius * np.sin(t_normalized)
        dy = radius * np.cos(t_normalized)
        yaw = np.arctan2(dy, dx)

        quat = R.from_euler('zyx', [yaw, 0, 0]).as_quat()

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

    def generate_and_publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "world"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        t = 0.0
        while t <= self.total_time:
            pose = self.path(t)
            pose.header.stamp = self.get_clock().now().to_msg()
            path_msg.poses.append(pose)
            t += self.dt

        # Wait for subscribers
        while self.path_publisher.get_subscription_count() == 0:
            self.get_logger().warn('Waiting for subscriber to full_path...')
            rclpy.spin_once(self, timeout_sec=0.1)

        self.path_publisher.publish(path_msg)
        self.get_logger().info('Full path published.')


    def request_callback(self, msg: Float32):
        time_request = msg.data
        desired_pose = self.path(time_request)
        desired_pose.header.stamp = self.get_clock().now().to_msg()
        self.pose_publisher.publish(desired_pose)
        # self.get_logger().info(f'Published desired pose for time {time_request:.2f} seconds.')


def main(args=None):
    rclpy.init(args=args)
    node = PathGeneration()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
