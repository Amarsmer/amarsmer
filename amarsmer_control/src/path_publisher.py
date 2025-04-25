#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher = self.create_publisher(Path, 'set_path', 10)
        self.timer = self.create_timer(1.0, self.publish_path)

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "world"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # Path parameters
        radius = 2.0
        depth_per_circle = 1.0
        num_turns = 3
        steps_per_turn = 100
        total_steps = num_turns * steps_per_turn
        z_start = 0.0

        # Generate the path
        t_vals = np.linspace(0, 2 * np.pi * num_turns, total_steps)
        x_vals = radius * np.cos(t_vals)
        y_vals = radius * np.sin(t_vals)
        z_vals = z_start - (depth_per_circle * t_vals) / (2 * np.pi)

        for i in range(total_steps):
            x, y, z = x_vals[i], y_vals[i], z_vals[i]
            dx = -radius * np.sin(t_vals[i])
            dy = radius * np.cos(t_vals[i])
            yaw = np.arctan2(dy, dx)
            quat = R.from_euler('zyx', [yaw, 0, 0]).as_quat()

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

            path_msg.poses.append(pose)

        self.publisher.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()