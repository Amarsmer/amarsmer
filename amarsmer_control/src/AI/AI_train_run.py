#!/usr/bin/env python3

# rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy

# Common python libraries
import time
import numpy as np
from datetime import datetime

# ROS2 msg libraries
from std_msgs.msg import String, Float32, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker

# Custom libraries
from amarsmer_control import ROV
import functions as f

# Training specific custom librairies
from amarsmer_control import ROV
from backprop import NN
from online_training import PyTorchOnlineTrainer
from monitoring import RobotMonitorAdapter

# Training specific librairies
import torch
import json
import threading
import atexit

class Controller(Node):
    def __init__(self):

        super().__init__('ai_control', namespace='amarsmer')

        # self.declare_parameter('name', 'data')
        self.declare_parameter('display', True)
        self.declare_parameter('load_weights', False)
        self.declare_parameter('train', True)
        self.declare_parameter('continue_running', True)
        self.declare_parameter('target', '0 0 0 0 0 0')
        self.input_string = ''

        self.rov = ROV(self, thrust_visual = True)

        self.odom_subscriber = self.create_subscription(Odometry, '/amarsmer/odom', self.odom_callback, 10)
        self.str_input_subscriber = self.create_subscription(String, '/amarsmer/input_str', self.str_input_callback, 10)
        self.pose_arrow_publisher = self.create_publisher(Marker, "/pose_arrow", 10)

        self.data_publisher = self.create_publisher(Float32MultiArray, "/monitoring_data", 10)

        self.future = None # Used for client requests

        self.timer = self.create_timer(0.1, self.run)

        ## Initiating variables

        # Définir les dimensions du monde pour le monitoring
        self.WORLD_BOUNDS = (-10, 10, -10, 10)  # x_min, x_max, y_min, y_max

        # Configuration de base
        HL_size = 800
        input_size = 6
        output_size = 2

        self.Q_weight = np.diag([50, # x
                                 50, # y 
                                 40, # psi
                                 1, # u
                                 1, # v
                                 1  # r
                                 ])
        
        self.R_weight = np.diag([0.03, # X
                                 0.03, # Y
                                 0.03   # N
                                 ])

        # Create pytorch network
        self.network = NN(input_size, HL_size, output_size)

        self.trainer = None
        self.training_initiated = False

        # Initiate monitoring data
        self.monitoring = []
        self.monitoring.append(['x','y','psi','x_d','y_d','psi_d','u1','u2','t'])

        self.date = datetime.today().strftime('%Y_%m_%d-%H_%M_%S')

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def odom_callback(self, msg: Odometry):
        pose, twist = f.odometry(msg)

        self.rov.current_pose = pose
        self.rov.current_twist = twist

        # self.get_logger().info(f"{self.pose}")
        # self.get_logger().info(f"{self.twist}")
    
    def str_input_callback(self, msg: String):
        self.input_string = msg.data

    def run(self):
        if not self.rov.ready():
            return
        
        target = self.get_parameter('target').get_parameter_value().string_value
        target = list(map(float, target.split())) # convert a multiple values string to a list

        target_pose = f.make_pose(target)
        f.create_pose_marker(target_pose, self.pose_arrow_publisher)

        if not self.training_initiated:
            self.training_initiated = True

            self.t0 = self.get_time()

            # Initialize the robot
            # self.monitor = RobotMonitorAdapter(world_bounds=self.WORLD_BOUNDS)

            # Ask for display preference
            # self.display_curves= self.get_parameter('display').get_parameter_value().bool_value
            # self.get_logger().info(f"stored display bool: {self.display_curves}")

            # if self.display_curves:
            #     self.monitor.start_monitoring()
            #     time.sleep(1)

            # Weight loading
            if self.get_parameter('load_weights').get_parameter_value().bool_value:
                with open('last_w_torch.json') as fp:
                    json_obj = json.load(fp)
                self.network.load_weights_from_json(json_obj, HL_size)
                

            # Init PyTorch trainer with monitoring
            # monitor_instance = self.monitor if self.display_curves else None
            self.trainer = PyTorchOnlineTrainer(self.rov, self.network, None, self.Q_weight, self.R_weight)

            train = self.get_parameter('train').get_parameter_value().bool_value #Boolean

            if self.rov.current_pose == None: # Make sure the robot sim is loaded, possibly redundant with rov.ready()
                return
            self.trainer.training = (train)

            # Main training loop, currently runs only once
            continue_running = True
            session_count = 0
            session_count += 1

            self.get_logger().info(f"\n⚙️ Starting training session #{session_count}")

            self.training_thread = threading.Thread(target=self.trainer.train, args=(target,)) # Start training process on a separate thread
            self.trainer.running = True
            self.training_thread.start()
            
            #TODO: add flexibility to run multiple training sessions
            """
            input_string = self.get_parameter('input_string').value
            try:
                if input_string != '':
                    # self.input_string = ''
                    self.set_parameters([Parameter('input_string', Parameter.Type.STRING, '')])
                    # input("Press Enter to stop the current training")
                    self.trainer.running = False
                
                training_thread.join(timeout=5)
                if training_thread.is_alive():
                    print("⚠️ Training thread did not finish in time, continuing anyway")
                    
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Stopping training...")
                self.trainer.running = False
                training_thread.join(timeout=5)

            if display_curves:
                monitor.save_results(f"session_{session_count}_{time.strftime('%Y%m%d_%H%M%S')}")

            # choice = ''
            self.get_logger().info("Do you want to continue? (y/n) --> ")
            while input_string.lower() not in ['y', 'n']:
                input_string = self.get_parameter('input_string').value
                self.get_logger().info(f"stored variable: {input_string}")
                # choice = input("Do you want to continue? (y/n) --> ")

            if input_string.lower() == 'y':
                choice_learning = ''
                while choice_learning.lower() not in ['y', 'n']:
                    choice_learning = input('Do you want to learn? (y/n) --> ')
                
                self.trainer.training = (choice_learning.lower() == 'y')
                
                # target_input = input("Move the robot to the initial point and enter the new target : x y radian --> ")
                # target = [float(x) for x in target_input.split()]
                # if len(target) != 3:
                #     raise ValueError("Need exactly 3 values")
            
            else:
                continue_running = False
            """

        
        # self.get_logger().info(f"Grad: {self.trainer.gradient_flag}") # Print gradient for debugging purposes

        ### Save data for monitoring
        if self.trainer.command_set: # Make sure the training has started
            x_m = self.rov.current_pose[0]
            y_m = self.rov.current_pose[1]
            psi_m = self.rov.current_pose[5]

            x_d_m = target[0]
            y_d_m = target[1]
            psi_d_m = target[2]

            u = self.trainer.command

            t = self.get_time() - self.t0

            data_array = [x_m, y_m, psi_m, x_d_m, y_d_m , psi_d_m, u[0],u[1], t]

            self.monitoring.append(data_array)

            publisher_msg = Float32MultiArray()
            publisher_msg.data = data_array
            self.data_publisher.publish(publisher_msg)
            # self.get_logger().info(f'Publishing: {msg.data}')

        if self.input_string == 'stop': # Stop training session from terminal
            self.input_string = ''
            self.trainer.running = False
            self.training_thread.join(timeout=5)
            # if self.display_curves:
            #     # self.monitor.save_results(f"session_{session_count}_{time.strftime('%Y%m%d_%H%M%S')}")
            #     self.monitor.save_results(f"session_1_{time.strftime('%Y%m%d_%H%M%S')}")

            title = 'data/AI_data/' + self.date +'-AI_data'
            np.save(title, self.monitoring)

            # Save the weights
            json_obj = self.network.save_weights_to_json()
            with open('last_w_torch.json', 'w') as fp:
                json.dump(json_obj, fp)


            # if self.display_curves:
            #     self.monitor.save_results(f"final_results_{time.strftime('%Y%m%d_%H%M%S')}")
            #     self.monitor.stop_monitoring()
            # else:
            #     print("⚠️ No results saved, monitoring was disabled")
            self.get_logger().info("Training stopped")

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()