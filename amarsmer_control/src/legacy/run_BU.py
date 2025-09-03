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

import torch
import json
import threading
import atexit
import time
# from robot_sim import ZMQPioneerSimulation
from amarsmer_control import ROV
from backprop import PioneerNN
from online_training import PyTorchOnlineTrainer
from monitoring import RobotMonitorAdapter

class Controller(Node):
    def __init__(self):

        super().__init__('mpc_control', namespace='amarsmer')

        self.rov = ROV(self, thrust_visual = True)

        self.odom_subscriber = self.create_subscription(Odometry, '/amarsmer/odom', self.odom_callback, 10)
        self.pose_arrow_publisher = self.create_publisher(Marker, "/pose_arrow", 10)

        # Create a client for path request
        self.client = self.create_client(RequestPath, '/path_request')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")
        
        self.future = None # Used for client requests

        self.timer = self.create_timer(0.01, self.run)

        ## Initiating variables

        # Pose
        self.current_pose = None
        self.current_twist = None

        # Définir les dimensions du monde pour le monitoring
        self.WORLD_BOUNDS = (-10, 10, -10, 10)  # x_min, x_max, y_min, y_max

        self.r = 0.096  # wheel radius
        self.R = 0.267  # demi-distance entre les roues
        '''
        # MPC Parameters
        self.mpc_horizon = 1
        self.mpc_time = 1.2
        self.mpc_path = Path()
        self.input_bounds = {"lower": np.array([-40.0, -15.0]),
                             "upper": np.array([40.0, 15.0]),
                             "idx":   np.array([0, 1])
                             }
        self.Q_weight = np.diag([50, # x
                                 50, # y 
                                 50, # psi
                                 20, # u
                                 20  # r
                                 ])

        self.R_weight = np.diag([0.1, # X
                                 0.5  # N
                                 ])

        # Initialize MPC solver
        self.controller = None #Updated at the start of spin

        # Initialize monitoring values
        self.monitoring = []
        self.monitoring.append(['x','y','psi','x_d','y_d','psi_d','u1','u2','t'])

        self.date = datetime.today().strftime('%Y_%m_%d-%H_%M_%S')
        '''

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def pose_to_array(self, msg_pose): # Used to convert pose msg to a regular array
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

        return [x,y,z,roll,pitch,yaw]

    def odom_callback(self, msg: Odometry):
        # Extract pose
        msg_pose = msg.pose.pose

        self.current_pose = self.pose_to_array(msg_pose)

        # Extract twist
        twist = msg.twist.twist
        u = twist.linear.x
        v = twist.linear.y
        w = twist.linear.z

        p = twist.angular.x
        q = twist.angular.y
        r = twist.angular.z
        
        self.current_twist = [u,v,w,p,q,r]

        # self.get_logger().info(f"{self.pose}")
        # self.get_logger().info(f"{self.twist}")

    def create_pose_marker(self, inPose):
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

        self.pose_arrow_publisher.publish(marker)

    def run(self):
        if not self.rov.ready():
            return

        # Initialize the robot
        monitor = RobotMonitorAdapter(world_bounds=self.WORLD_BOUNDS)
        trainer = None

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

        # Configuration de base  (par défaut pour la couche cachée.Peut etre modifié dans le module torch_back.py)
        HL_size = 10
        input_size = 3
        output_size = 2

        # Création du réseau PyTorch
        network = PioneerNN(input_size, HL_size, output_size)

        # Ask for display preference
        display_choice = ''
        while display_choice.lower() not in ('y', 'n'):
            display_choice = input('Enable real-time display? (y/n) --> ')

        if display_choice.lower() == 'y':
            monitor.start_monitoring()

        # Chargement de poids existants
        choice = input('Do you want to load previous network? (y/n) --> ')
        if choice == 'y':
            with open('last_w_torch.json') as fp:
                json_obj = json.load(fp)
            network.load_weights_from_json(json_obj, HL_size)
            

        # Initialiser le trainer PyTorch avec monitoring
        monitor_instance = monitor if display_choice.lower() == 'y' else None
        trainer = PyTorchOnlineTrainer(self.rov, network, monitor_instance)

        choice = ''
        while choice!='y' and choice !='n':
            choice = input('Do you want to learn? (y/n) --> ')

        trainer.training = (choice.lower() == 'y')

        # Demander la cible
        target_input = input("Enter the first target : x y radian --> ")
        target = target_input.split()
        if len(target) != 3:
            raise ValueError("Need exactly 3 values")
        for i in range(len(target)):
            target[i] = float(target[i])

        # TODO self.create_pose_marker(desired_pose)


        # Boucle principale d'entraînement
        continue_running = True
        session_count = 0

        while continue_running:
            session_count += 1
            print(f"\n⚙️ Starting training session #{session_count}")

            thread = threading.Thread(target=trainer.train, args=(target,))
            trainer.running = True
            thread.start()

            try:
                input("Press Enter to stop the current training")
                trainer.running = False
                
                thread.join(timeout=5)
                if thread.is_alive():
                    print("⚠️ Training thread did not finish in time, continuing anyway")
                    
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Stopping training...")
                trainer.running = False
                thread.join(timeout=5)

            if display_choice.lower() == 'y':
                monitor.save_results(f"session_{session_count}_{time.strftime('%Y%m%d_%H%M%S')}")

            choice = ''
            while choice.lower() not in ['y', 'n']:
                choice = input("Do you want to continue? (y/n) --> ")

            if choice.lower() == 'y':
                choice_learning = ''
                while choice_learning.lower() not in ['y', 'n']:
                    choice_learning = input('Do you want to learn? (y/n) --> ')
                
                trainer.training = (choice_learning.lower() == 'y')
                
                target_input = input("Move the robot to the initial point and enter the new target : x y radian --> ")
                target = [float(x) for x in target_input.split()]
                if len(target) != 3:
                    raise ValueError("Need exactly 3 values")
                
            else:
                continue_running = False

        # Save the weights
        json_obj = network.save_weights_to_json()
        with open('last_w_torch.json', 'w') as fp:
            json.dump(json_obj, fp)


        if display_choice.lower() == 'y':
            monitor.save_results(f"final_results_{time.strftime('%Y%m%d_%H%M%S')}")
        else:
            print("⚠️ No results saved, monitoring was disabled")

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()