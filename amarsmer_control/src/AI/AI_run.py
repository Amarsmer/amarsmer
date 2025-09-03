#!/usr/bin/env python3

# rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy

# Common python libraries
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime

# ROS2 msg libraries
from std_msgs.msg import String, Float32, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker

# Custom libraries
from urdf_parser_py import urdf
from hydrodynamic_model import hydrodynamic
import ur_mpc
from amarsmer_control import ROV
from amarsmer_interfaces.srv import RequestPath
import functions as f

import torch
import json
import threading
import atexit
import time
# from robot_sim import ZMQPioneerSimulation
from amarsmer_control import ROV
from backprop import NN
from online_training import PyTorchOnlineTrainer
from monitoring import RobotMonitorAdapter

class Controller(Node):
    def __init__(self):

        super().__init__('mpc_control', namespace='amarsmer')

        # self.declare_parameter('name', 'data')
        self.declare_parameter('display', False)
        self.declare_parameter('load_weights', False)
        self.declare_parameter('train', True)
        self.declare_parameter('target', '5 0 0')


        self.rov = ROV(self, thrust_visual = True)

        self.odom_subscriber = self.create_subscription(Odometry, '/amarsmer/odom', self.odom_callback, 10)
        self.pose_arrow_publisher = self.create_publisher(Marker, "/pose_arrow", 10)
        """
        # Create a client for path request
        self.client = self.create_client(RequestPath, '/path_request')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")
        """
        self.future = None # Used for client requests

        self.timer = self.create_timer(0.01, self.run)

        ## Initiating variables

        # Définir les dimensions du monde pour le monitoring
        self.WORLD_BOUNDS = (-10, 10, -10, 10)  # x_min, x_max, y_min, y_max

        self.r = 0.096  # wheel radius
        self.R = 0.267  # demi-distance entre les roues

        # Configuration de base
        HL_size = 10
        input_size = 3
        output_size = 2

        # Création du réseau PyTorch
        self.network = NN(input_size, HL_size, output_size)

        self.trainer = None

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def odom_callback(self, msg: Odometry):
        pose, twist = f.odometry(msg)

        self.rov.current_pose = pose
        self.rov.current_twist = twist

        # self.get_logger().info(f"{self.pose}")
        # self.get_logger().info(f"{self.twist}")

    def run(self):
        if not self.rov.ready():
            return

        # Initialize the robot
        monitor = RobotMonitorAdapter(world_bounds=self.WORLD_BOUNDS)

        '''
        # Register cleanup function to ensure simulation is stopped
        def cleanup():
            print("Cleaning up and stopping simulation...")
            if hasattr(robot, 'cleanup'):
                robot.cleanup()
            if hasattr(monitor, 'stop_monitoring'):
                monitor.stop_monitoring()
        atexit.register(cleanup)
        '''

        # Wait a moment to ensure the simulation is fully started
        # time.sleep(1)

        # Ask for display preference
        display_curves= self.get_parameter('display').get_parameter_value().bool_value

        if display_curves:
            monitor.start_monitoring()

        # Chargement de poids existants
        if self.get_parameter('load_weights').get_parameter_value().bool_value and False:
            with open('last_w_torch.json') as fp:
                json_obj = json.load(fp)
            self.network.load_weights_from_json(json_obj, HL_size)
            

        # Initialiser le trainer PyTorch avec monitoring
        monitor_instance = monitor if display_curves else None
        self.trainer = PyTorchOnlineTrainer(self.rov, self.network, monitor_instance)

        train = self.get_parameter('load_weights').get_parameter_value().bool_value #Boolean

        if self.rov.current_pose == None:
            return
        self.trainer.training = (train)

        """
        # Demander la cible
        target_input = input("Enter the first target : x y radian --> ")
        target = target_input.split()
        if len(target) != 3:
            raise ValueError("Need exactly 3 values")
        for i in range(len(target)):
            target[i] = float(target[i])
        """
        target = self.get_parameter('target').get_parameter_value().string_value
        target = list(map(float, target.split())) # convert a multiple values string to a list

        target_pose = f.make_pose(target)
        f.create_pose_marker(target_pose, self.pose_arrow_publisher)



        # Boucle principale d'entraînement
        continue_running = True
        session_count = 0

        while continue_running:
            session_count += 1
            print(f"\n⚙️ Starting training session #{session_count}")

            thread = threading.Thread(target=self.trainer.train, args=(target,))
            self.trainer.running = True
            thread.start()

            try:
                input("Press Enter to stop the current training")
                self.trainer.running = False
                
                thread.join(timeout=5)
                if thread.is_alive():
                    print("⚠️ Training thread did not finish in time, continuing anyway")
                    
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Stopping training...")
                self.trainer.running = False
                thread.join(timeout=5)

            if display_curves:
                monitor.save_results(f"session_{session_count}_{time.strftime('%Y%m%d_%H%M%S')}")

            choice = ''
            while choice.lower() not in ['y', 'n']:
                choice = input("Do you want to continue? (y/n) --> ")

            if choice.lower() == 'y':
                choice_learning = ''
                while choice_learning.lower() not in ['y', 'n']:
                    choice_learning = input('Do you want to learn? (y/n) --> ')
                
                self.trainer.training = (choice_learning.lower() == 'y')
                
                target_input = input("Move the robot to the initial point and enter the new target : x y radian --> ")
                target = [float(x) for x in target_input.split()]
                if len(target) != 3:
                    raise ValueError("Need exactly 3 values")
                
            else:
                continue_running = False

        # Save the weights
        json_obj = self.network.save_weights_to_json()
        with open('last_w_torch.json', 'w') as fp:
            json.dump(json_obj, fp)


        if display_curves:
            monitor.save_results(f"final_results_{time.strftime('%Y%m%d_%H%M%S')}")
        else:
            print("⚠️ No results saved, monitoring was disabled")

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()