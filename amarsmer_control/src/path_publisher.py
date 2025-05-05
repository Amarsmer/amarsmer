#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time

# Subscribes to path_generation and saves the full path msg, then publishes it to rviz

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher = self.create_publisher(Path, 'set_path', 10)
        self.timer = self.create_timer(1.0, self.publish_path)

        self.saved_path = Path()

        self.create_subscription(Path,'full_path', self.save_path, 10)

    def save_path(self, msg: Path):
        self.get_logger().info(f'Received path with {len(msg.poses)} poses')
        self.saved_path = msg
        self.publisher.publish(self.saved_path)

    def publish_path(self):
        self.publisher.publish(self.saved_path)


def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()