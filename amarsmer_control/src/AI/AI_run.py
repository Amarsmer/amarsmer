#!/usr/bin/env python3

# rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy

# Common python libraries
import numpy as np
from ament_index_python.packages import get_package_share_directory

# ROS2 msg libraries
from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose

# Custom libraries
import custom_functions as cf
from amarsmer_interfaces.msg import InCtrl

# Training specific custom librairies
from backprop import NN
from online_training import PyTorchOnlineTrainer

# Training specific librairies
import json
import threading

class Controller(Node):
    def __init__(self):

        super().__init__('ai_control', namespace='amarsmer')

        self.declare_parameter('network_name', '')       # Will load a saved network file if specified, otherwise will initialize one
        self.declare_parameter('train', True)            # Wether network are updated, 'False' implies testing
        self.declare_parameter('automate', True)         # Test automation will change the robot's pose if some loss requirements are met
        self.input_string = ['','']                      # Meant to be used as ['instruction', 'argument']
        self.declare_parameter('dt', 0.05)

        #################################### ROS2 Communication ####################################

        self.aiData_publisher = self.create_publisher(Float32MultiArray, "/amarsmer/aiData",10)
        self.thruster_input_publisher = self.create_publisher(Float32MultiArray, "/thruster_input",10)
        self.InCtrl_subscriber = self.create_subscription(InCtrl, '/amarsmer/InCtrl', self.ctrl_callback, 10)
        
        self.dt = self.get_parameter('dt').get_parameter_value().double_value # Used both for run and pose computation
        self.timer = self.create_timer(self.dt, self.run)

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
        
        #################################### Initiating variables ####################################

        self.state = None
        self.ai_path = Path()
        
        # Weighting matrices
        self.Q_weight = np.diag([50, # x
                                 50, # y 
                                 10, # psi
                                 1, # u
                                 1, # v
                                 1  # r
                                 ])
        
        self.R_weight = np.diag([1e-5, # u1
                                 1e-5  # u2
                                 ])
        # TODO The difference of R weight between AI and MPC may come from delta_t applied to the gradient, further investigation required

        ### Create pytorch network

        # Network parameters
        self.HL_size = 40
        input_size = 6 # x, y, psi, u, v, r
        output_size = 2 # u1, u2
        self.learning_rate = 1e-4

        self.trainer = None
        self.training_initiated = False

        # network loading
        network_name = self.get_parameter('network_name').get_parameter_value().string_value
        if network_name == '':
            self.get_logger().info(f"No network loaded. Initializing random network weights with hidden layer size: {self.HL_size}.")
            self.network = NN(input_size, self.HL_size, output_size)

        else:
            # Check if the file exists
            try:
                with open(f'saved_networks/{network_name}.json') as fp:
                    json_obj = json.load(fp)
                
            # If it does not, display error message and create a new network
            except:
                self.get_logger().info(f"#################### ERROR: no network file with the name: {network_name}. Initializing random network with hidden layer size: {self.HL_size}. ####################")
                self.network = NN(input_size, self.HL_size, output_size)

            # If it exists, adjust the hidden layer size and load the network
            else:
                self.HL_size = len(json_obj["input_weights"][0][:])
                self.network = NN(input_size, self.HL_size, output_size)
                self.network.load_network_from_json(json_obj)
                self.get_logger().info(f"Loading network json: {network_name}.")

        ### Test automation
        # If the loss is below the threshold for the entire set delay, the robot is moved to the next pose
        self.automate = self.get_parameter('automate').get_parameter_value().bool_value
        if self.automate:
            self.loss_threshold = 15.
            self.previous_loss = 1e10 # Initialized at an arbitrarily high value instead of None to reduce the number of if statements
            self.minimal_loss_timer = None
            self.acceptable_loss_delay = 10. # If the loss is below the threshold for this amount of time (s), the robot is moved
            
            # Poses to be parsed 
            self.pose_index = 0
            self.initial_poses = [np.array([4., 4., 0., 0., 0., 1.]),
                                np.array([-4., -4., 0., 0., 0., 4.]),
                                np.array([4., -4., 0., 0., 0., 1.]),
                                np.array([-4., 4., 0., 0., 0., np.pi/2]),
                                np.array([0., -4., 0., 0., 0., np.pi/2.]),
                                np.array([-4., 4., 0., 0., 0., -np.pi/2]),
                                np.array([0., -4., 0., 0., 0., -np.pi/2.]),
                                np.array([15., 4., 0., 0., 0., 1.]),
                                np.array([-4., 15., 0., 0., 0., 4.]),
                                np.array([15., 15., 0., 0., 0., 0.])]

        self.current_time = self.get_time()

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def ctrl_callback(self, msg):
        self.state = np.array(msg.state.data).reshape(-1, 1)
        self.ai_path = msg.path
    
    def str_input_callback(self, msg: String):
        self.input_string = msg.data.split()

    def run(self):
        # Update time
        self.current_time = self.get_time()

        #################################### Initialize ####################################

        if not self.training_initiated: # This code used to be in a while loop and requires adjustements to work as a ROS2 node
            self.training_initiated = True

            self.t0 = self.get_time() # Initial time for data collection
                    
            # Initialize trainer
            self.trainer = PyTorchOnlineTrainer(self.network, self.learning_rate, self.Q_weight, self.R_weight)

            train = self.get_parameter('train').get_parameter_value().bool_value #Boolean

            # Main training
            self.target = [0.,0.,0.,0.,0.,0.] # Default initial target

            self.trainer.updateTarget(self.target) # To be used for trajectory tracking

            self.get_logger().info(f"\n Starting training session")

            self.training_thread = threading.Thread(target=self.trainer.train, args=(self.target,)) # Start training process on a separate thread
            self.trainer.running = True
            self.training_thread.start()

        if self.ai_path.poses and self.state is not None: # Make sure the path is not empty

            self.target = cf.compute_target(self.ai_path, self.dt)
            self.trainer.updateTarget(self.target)

            self.trainer.updateState(self.state)

        #################################### Training automation ####################################

        if self.trainer.loss and self.automate: # Make sure the loss has been initialized

            # Detect when loss is below threshold (edge detection)
            if self.trainer.loss < self.loss_threshold :
                if self.previous_loss >= self.loss_threshold :
                    self.minimal_loss_timer = self.current_time
            else:
                self.minimal_loss_timer = None # Prevents false positives of robot briefly gets below threshold

            # Change robot's pose after loss remains under threshold for a set time
            if self.minimal_loss_timer and (self.current_time - self.minimal_loss_timer) > self.acceptable_loss_delay:
                    cf.set_pose_gz(self.initial_poses[self.pose_index])
                    self.pose_index += 1
                    self.pose_index %= len(self.initial_poses) # Makes sure the index wraps around instead of getting outside of the list
                    self.minimal_loss_timer = None
                    self.get_logger().info(f"\n Loss requirements met. Current pose index: {self.pose_index}")

            self.previous_loss = self.trainer.loss

        #################################### Update robot control and publish it to main control node ####################################
        self.u = self.trainer.u.ravel()

        publisher_msg = Float32MultiArray()
        publisher_msg.data = self.u
        self.thruster_input_publisher.publish(publisher_msg)

        # 'grad1', 'grad2', 'loss_x', 'loss_u'

        ## Publish AI specific data
        if not self.trainer.trainer_set:
            return

        grad = self.trainer.gradient_display.ravel()
        loss = self.trainer.loss_display.ravel()

        AI_data = [*grad,*loss]

        publisher_msg = Float32MultiArray()
        publisher_msg.data = AI_data
        self.aiData_publisher.publish(publisher_msg)

        # Debug info
        # self.get_logger().info(f"Grad: {self.trainer.gradient_display}") 
            
        #################################### Stop training and record data ####################################
        
        if self.input_string[0] == 'stop': # Stop training session from terminal, there is currently no way to restart training
            network_name = self.input_string[1]
            self.input_string = ['','']
            self.trainer.running = False
            self.training_thread.join(timeout=5)

            # Save the network
            json_obj = self.network.save_network_to_json()
            with open(f'saved_networks/{network_name}.json', 'w') as fp:
                
                json.dump(json_obj, fp)

            self.get_logger().info("Training stopped")

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()