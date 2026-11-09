#!/usr/bin/env python3

# rclpy
from rclpy.node import Node
import rclpy

# Common python libraries
import numpy as np

# ROS2 msg libraries
from std_msgs.msg import String, Float32MultiArray

# Custom libraries
import custom_functions as cf
from amarsmer_interfaces.msg import InCtrl

# Training specific custom librairies
from backprop import NN
from online_training import PyTorchOnlineTrainer

# Training specific librairies
import json


class Controller(Node):
    def __init__(self):

        super().__init__('ai_control', namespace='amarsmer')

        self.declare_parameter('network_name', '')       # Will load a saved network file if specified, otherwise will initialize one
        self.declare_parameter('train', True)            # Whether the network is updated, 'False' implies testing
        self.declare_parameter('automate', '')           # Test automation will change the robot's pose if some loss requirements are met
        self.input_string = ['', '']                     # Meant to be used as ['instruction', 'argument']
        self.declare_parameter('nb_thrusters', 2)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('order', 1)

        self.nb_thrusters = self.get_parameter('nb_thrusters').get_parameter_value().integer_value
        self.order = self.get_parameter('order').get_parameter_value().integer_value

        ####################################
        #
        # ROS2 Communication
        #
        ####################################

        self.aiData_publisher = self.create_publisher(Float32MultiArray, "/amarsmer/ctrlData", 10)
        self.thruster_input_publisher = self.create_publisher(Float32MultiArray, "/thruster_input", 10)

        self.InCtrl_subscriber = self.create_subscription(InCtrl, '/amarsmer/InCtrl', self.ctrl_callback, 10)
        self.str_subscriber = self.create_subscription(String, '/amarsmer/input_str', self.str_input_callback, 10)

        self.dt = self.get_parameter('dt').get_parameter_value().double_value  # Used both for run and gradient computation
        self.timer = self.create_timer(self.dt, self.run)

        ####################################
        #
        # Initiating variables
        #
        ####################################

        self.training_done = False

        self.state = None
        self.target = None
        self.ai_path = None
        self.new_state_available = False  # Set by ctrl_callback, consumed by run
        self.u = np.zeros(self.nb_thrusters)

        # Weighting matrices
        self.Q_weight = np.diag([5,  # x
                                 20,  # y
                                 100,  # cos psi
                                 100,  # sin psi
                                 1,   # u
                                 1,   # v
                                 1    # r
                                 ])

        self.R_weight = np.diag([0.015]*self.nb_thrusters)

        ### Create pytorch network

        # Network parameters
        self.HL_size = 40
        input_size = 7*self.order  # x, y, cos psi, sin psi, u, v, r
        output_size = self.nb_thrusters  # u1, u2, ...
        self.learning_rate = 1e-3

        network_name = self.get_parameter('network_name').get_parameter_value().string_value
        # network loading
        if network_name == '':
            self.get_logger().info(f"No network loaded. Initializing random network weights with hidden layer size: {self.HL_size}.")
            self.network = NN(input_size, self.HL_size, output_size)

        else:
            # Check if the file exists
            try:
                with open(f'saved_networks/{network_name}.json') as fp:
                    json_obj = json.load(fp)

            # If it does not, display error message and create a new network
            except Exception:
                self.get_logger().info(f"#################### ERROR: no network file with the name: {network_name}. Initializing random network with hidden layer size: {self.HL_size}. ####################")
                self.network = NN(input_size, self.HL_size, output_size)

            # If it exists, adjust the hidden layer size and load the network
            else:
                self.HL_size = len(json_obj["input_weights"][0][:])
                self.network = NN(input_size, self.HL_size, output_size)
                self.network.load_network_from_json(json_obj)
                self.get_logger().info(f"Loading network json: {network_name}.")

        ####################################
        #
        # Trainer
        #
        ####################################

        self.train = self.get_parameter('train').get_parameter_value().bool_value
        self.trainer = PyTorchOnlineTrainer(self.network,
                                            self.learning_rate,
                                            self.order,
                                            self.Q_weight,
                                            self.R_weight,
                                            self.nb_thrusters,
                                            dt=self.dt)
        self.trainer.training = self.train

        self.t0 = self.get_time()  # Initial time for data collection
        self.get_logger().info("\n Starting session")
        self.get_logger().info(f"\n Training is currently set to: {self.train}")

        ####################################
        #
        # Test automation
        #
        ####################################
        # If the loss is below the threshold for the entire set duration, the robot is moved to the next pose
        self.automate = self.get_parameter('automate').get_parameter_value().string_value
        self.automate_valid = False

        if self.automate in ['random', 'poses', 'circle']: # Check if the keyword is valid
            self.automate_valid = True
            self.loss_threshold = 2
            self.previous_loss = 1e10  # Initialized at an arbitrarily high value instead of None to reduce the number of if statements
            self.minimal_loss_timer = None
            self.acceptable_loss_delay = 10.  # If the loss is below the threshold for this amount of time (s), the robot is moved

            # Poses to be parsed
            self.pose_index = 0
            if self.automate == 'poses':
                self.initial_poses = [np.array([3., 3., 0., 0., 0., 1.]),
                                      np.array([-3., -3., 0., 0., 0., 4.]),
                                      np.array([4., -4., 0., 0., 0., 1.]),
                                      np.array([-4., 4., 0., 0., 0., np.pi/2]),
                                      np.array([0., -4., 0., 0., 0., np.pi/2.]),
                                      np.array([-5., 5., 0., 0., 0., -np.pi/2]),
                                      np.array([0., -5., 0., 0., 0., -np.pi/2.]),
                                      np.array([5., 0., 0., 0., 0., 0.]),
                                      np.array([-5., 0., 0., 0., 0., 0.])]

            if self.automate == 'circle':
                angles = np.arange(12)*np.pi/6
                self.initial_poses = []
                for i in range(len(angles)):
                    d = 5
                    x = d * np.cos(angles[i]) #*(-1)**i
                    y = d * np.sin(angles[i]) #*(-1)**i
                    psi = np.arctan(y/x)
                    self.initial_poses.append(np.array([x, y, 0., 0., 0., psi]))

        else:
            self.get_logger().info('ERROR: Automate keyword is not recognized, feature disabled for this run.')

        self.current_time = self.get_time()

    ####################################
    #
    # Functions
    #
    ####################################

    def random_pose(self, range = 4):
        state = self.state.ravel()

        random_vector = np.random.rand(1, 6)[0].ravel()*range*2 - range  # Randomize between -range and +range
        random_vector[2:5] = [0, 0, 0]

        return random_vector + state

    def get_time(self):
        s, ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def ctrl_callback(self, msg):
        self.state = np.array(msg.state.data).reshape(-1, 1)
        self.ai_path = msg.path
        self.new_state_available = True

    def str_input_callback(self, msg: String):
        self.input_string = msg.data.split()

    def run(self):
        self.current_time = self.get_time()

        ####################################
        #
        # Control + training step, exactly once per new state
        #
        ####################################

        if (self.new_state_available
                and self.state is not None
                and self.ai_path is not None
                and self.ai_path.poses):  # Make sure the path is not empty

            self.new_state_available = False

            self.target = cf.compute_target(self.ai_path, self.dt)
            self.u = self.trainer.step(self.state, self.target)

        ####################################
        #
        # Training automation
        #
        ####################################

        if self.trainer.loss is not None and self.automate_valid:  # Make sure the loss has been initialized

            # Detect when loss is below threshold (edge detection)
            if self.trainer.loss < self.loss_threshold:
                if self.previous_loss >= self.loss_threshold:
                    self.minimal_loss_timer = self.current_time
            else:
                self.minimal_loss_timer = None  # Prevents false positives if robot briefly gets below threshold

            # Change robot's pose after loss remains under threshold for a set time
            if self.minimal_loss_timer and (self.current_time - self.minimal_loss_timer) > self.acceptable_loss_delay:
                self.pose_index += 1
                if self.automate == 'random':
                    new_pose = self.random_pose()
                else:  # 'poses' or 'circle'
                    new_pose = self.initial_poses[self.pose_index % len(self.initial_poses)]

                    if self.pose_index == len(self.initial_poses):
                        self.trainer.training = False
                        self.training_done = True
                        self.get_logger().info("All initial training poses have been parsed. Training stopped")

                cf.set_pose_gz(new_pose)
                self.trainer.reset()  # Input history is no longer valid after a teleport

                self.minimal_loss_timer = None
                self.get_logger().info(f"\n Loss requirements met. Moving robot at {self.automate} ({self.pose_index}).")

            self.previous_loss = self.trainer.loss

        ####################################
        #
        # Publish control (last u is held between new states) and monitoring data
        #
        ####################################

        if not self.training_done:
            publisher_msg = Float32MultiArray()
            publisher_msg.data = [float(v) for v in self.u]
            self.thruster_input_publisher.publish(publisher_msg)

            if not self.trainer.trainer_set:
                return

            grad = self.trainer.gradient_display.ravel()
            loss = self.trainer.loss_display.ravel()

            publisher_msg = Float32MultiArray()
            publisher_msg.data = [float(v) for v in [*grad, *loss]]
            self.aiData_publisher.publish(publisher_msg)

        else:
            publisher_msg = Float32MultiArray()
            publisher_msg.data = [0.,0.]
            self.thruster_input_publisher.publish(publisher_msg)

        # Debug info
        # self.get_logger().info(f"Grad: {self.trainer.gradient_display}")
        # self.get_logger().info(f"Coordinates in target frame: \n{self.trainer.target_frame}")
        # self.get_logger().info(f"Skew: \n{self.trainer.skew_display}")

        ####################################
        #
        # Stop training and record network
        #
        ####################################

        if self.input_string[0] == 'stop':  # Stop training session from terminal
            network_name = self.input_string[1]
            self.input_string = ['', '']
            self.trainer.training = False  # No thread to join anymore, just stop updating

            # Save the network
            json_obj = self.network.save_network_to_json()
            robot = 'uvr' if self.nb_thrusters == 3 else 'ur'
            saved_name = f'{robot}_{self.order}-order_{network_name}'
            with open(f'saved_networks/{saved_name}.json', 'w') as fp:
                json.dump(json_obj, fp)

            self.get_logger().info(f"Training stopped, network saved as: {saved_name}")


rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()