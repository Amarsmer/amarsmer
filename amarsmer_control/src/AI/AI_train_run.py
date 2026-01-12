#!/usr/bin/env python3

# rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy

# Common python libraries
import time
import numpy as np
from datetime import datetime
import os

# ROS2 msg libraries
from std_msgs.msg import String, Float32, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker

# Custom libraries
from amarsmer_control import ROV
import custom_functions as cf

# Training specific custom librairies
from amarsmer_control import ROV
from backprop import NN
from online_training import PyTorchOnlineTrainer

# Training specific librairies
import torch
import json
import threading
import atexit

class Controller(Node):
    def __init__(self):

        super().__init__('ai_control', namespace='amarsmer')

        # self.declare_parameter('name', 'data')
        self.declare_parameter('load_weights', False)
        self.declare_parameter('train', True)
        self.declare_parameter('continue_running', True)
        self.declare_parameter('target', '0 0 0 0 0 0')
        self.input_string = ''

        self.rov = ROV(self, thrust_visual = True)

        # Publisher and subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/amarsmer/odom', self.odom_callback, 10)
        self.str_input_subscriber = self.create_subscription(String, '/amarsmer/input_str', self.str_input_callback, 10) # Used to interact with the training process
        self.pose_arrow_publisher = self.create_publisher(Marker, "/pose_arrow", 10)

        self.data_publisher = self.create_publisher(Float32MultiArray, "/monitoring_data", 10)
        self.network_publisher = self.create_publisher(Float32MultiArray, "/network_data", 10)

        self.timer = self.create_timer(0.1, self.run)

        ## Initiating variables
        # Network parameters
        HL_size = 40
        input_size = 6 # x, y, psi, u, v, r
        output_size = 2 # u1, u2
        self.learning_rate = 0.1
        self.momentum = 0.0001 # Portion of the gradient reported to the next one

        """
        ################# Reference weighting matrices meant to be equivalent to MPC

        # Weighting matrices
        self.Q_weight = np.diag([50, # x
                                 50, # y 
                                 40, # psi
                                 1, # u
                                 1, # v
                                 1  # r
                                 ])
        
        self.R_weight = np.diag([0.015, # u1
                                 0.015  # u2
                                 ])
        """

        # Weighting matrices
        self.Q_weight = np.diag([15, # x
                                 50, # y 
                                 40, # psi
                                 1, # u
                                 1, # v
                                 1  # r
                                 ])
        
        self.R_weight = np.diag([1e-5, # u1
                                 1e-5  # u2
                                 ])
        # TODO The difference of R weight between AI and MPC may come from delta_t applied to the gradient, further investigation required

        # Create pytorch network
        self.network = NN(input_size, HL_size, output_size)

        self.trainer = None
        self.training_initiated = False

        # Initiate monitoring data, both stored as .npy and published on a topic
        self.monitoring = []
        self.monitoring.append(['x','y','psi','x_d','y_d','psi_d','u1','u2', 'grad1', 'grad2', 'loss_X', 'loss_u', 'skew', 't']) # Naming the variables in the first row, useful as is and even more so if data points change between version

        self.date = datetime.today().strftime('%Y_%m_%d-%H_%M_%S')

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def odom_callback(self, msg: Odometry):
        pose, twist = cf.odometry(msg)

        self.rov.current_pose = pose
        self.rov.current_twist = twist
    
    def str_input_callback(self, msg: String):
        self.input_string = msg.data

    def updateRobotState(self):
        # Extract position and speed as column vectors
        position = np.array([self.rov.current_pose[x] for x in [0, 1, 5]]).reshape(-1, 1)
        speed = np.array([self.rov.current_twist[x] for x in [0, 1, 5]]).reshape(-1, 1)

        # Concatenate vertically (stack columns)
        self.state = np.vstack([position, speed])

        self.trainer.updateState(self.state)

    def run(self):
        if not self.rov.ready():
            return
        
        target = self.get_parameter('target').get_parameter_value().string_value
        target = list(map(float, target.split())) # convert a multiple values string to a list

        # Display target in gazebo
        target_pose = cf.make_pose(target)
        cf.create_pose_marker(target_pose, self.pose_arrow_publisher)

        # Initialize
        if not self.training_initiated: # This code used to be in a while loop and requires adjustements to work as a ROS2 node
            self.training_initiated = True

            self.t0 = self.get_time() # Initial time for data collection

            # Weight loading
            if self.get_parameter('load_weights').get_parameter_value().bool_value:
                dir_path = os.path.dirname(os.path.realpath(__file__))

                with open('/home/noe/ros2_ws/saved_weights/last_w_torch_MVP_momentum.json') as fp:
                    json_obj = json.load(fp)
                    HL_load_size = len(json_obj["input_weights"][0][:])
                    # self.get_logger().info(f"json = {HL_load_size}")
                self.network.load_weights_from_json(json_obj, HL_load_size)
                
            # Initialize trainer
            self.trainer = PyTorchOnlineTrainer(self.rov, self.network, self.learning_rate, self.momentum, self.Q_weight, self.R_weight)

            train = self.get_parameter('train').get_parameter_value().bool_value #Boolean

            # Main training loop, currently runs only once
            continue_running = True
            session_count = 0
            session_count += 1

            self.updateRobotState() # The trainer thread requires manual update of the robot's pose and twist
            self.trainer.updateTarget(target) # To be used for trajectory tracking

            self.get_logger().info(f"\n Starting training session #{session_count}")

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
                print("Training thread did not finish in time, continuing anyway")
                
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

        self.updateRobotState()

        ### Save and publish data for monitoring
        if self.trainer.command_set: # Make sure the training has started
            x_m = self.rov.current_pose[0]
            y_m = self.rov.current_pose[1]
            psi_m = self.rov.current_pose[5]

            x_d_m = target[0]
            y_d_m = target[1]
            psi_d_m = target[2]

            u = self.trainer.u.ravel()

            t = self.get_time() - self.t0

            grad = self.trainer.gradient_display.ravel()

            loss = self.trainer.loss.ravel()

            skew = self.trainer.skew

            data_array = [x_m, y_m, psi_m, x_d_m, y_d_m , psi_d_m, u[0],u[1], grad[0], grad[1], loss[0], loss[1], skew, t]

            self.monitoring.append(data_array)

            publisher_msg = Float32MultiArray()
            publisher_msg.data = data_array
            self.data_publisher.publish(publisher_msg)

            publisher_msg = Float32MultiArray()
            publisher_msg.data = self.trainer.error_display
            self.network_publisher.publish(publisher_msg)
            
            # Debug info
            # self.get_logger().info(f"Grad: {self.trainer.gradient_display}") 
            # self.get_logger().info(f"U: {self.trainer.u}") 
            # self.get_logger().info(f"Delta_t: {self.trainer.delta_t_display} \n") 
            # self.get_logger().info(f"\n Internal state: {self.trainer.state}")
            # self.get_logger().info(f"\n Internal error: {self.trainer.error}") 
            # self.get_logger().info(f"\n Train state: {self.trainer.state_train_display}")
            # self.get_logger().info(f"\n Train target: {self.trainer.target}") 
            # self.get_logger().info(f"\n Train error: {self.trainer.error_display}")
            # self.get_logger().info(f"\n Network input: {self.trainer.input_display}")
            # self.get_logger().info(f"\n Network input: {self.trainer.skew}")
            

        if self.input_string == 'stop': # Stop training session from terminal

            self.input_string = ''
            self.trainer.running = False
            self.training_thread.join(timeout=5)

            # Save the weights
            json_obj = self.network.save_weights_to_json()
            with open('last_w_torch.json', 'w') as fp:
                json.dump(json_obj, fp)

            self.get_logger().info("Training stopped")

            title = 'data/AI_data/' + self.date +'-AI_data'
            np.save(title, self.monitoring)

            

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()