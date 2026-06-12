#!/usr/bin/env python3

# rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import QoSDurabilityPolicy
import rclpy

# Common python libraries
import numpy as np
from datetime import datetime

# ROS2 msg libraries
from std_msgs.msg import String, Bool, Float32, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker

# Custom libraries
from urdf_parser_py import urdf
from amarsmer_interfaces.srv import RequestPath
from amarsmer_interfaces.msg import InCtrl
import custom_functions as cf

class Controller(Node):
    def __init__(self):

        super().__init__('control_manager', namespace='amarsmer')

        # Parameters
        self.declare_parameter('controller_type', 'MPC') 
        self.controller_type = self.get_parameter('controller_type').get_parameter_value().string_value

        self.declare_parameter('network_name', '') # Used by the AI controller, relevant here for data name
        network = self.get_parameter('network_name').get_parameter_value().string_value
        if network != '':
            network = f'_{network}'

        self.declare_parameter('comment', '')
        comment = self.get_parameter('comment').get_parameter_value().string_value
        if comment != '':
            comment = f'_{comment}'

        self.declare_parameter('simulation', True) 
        self.isSimulation = self.get_parameter('simulation').get_parameter_value().bool_value

        self.declare_parameter('dt', 0.05)
        self.dt = self.get_parameter('dt').get_parameter_value().double_value

        # Subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/amarsmer/odom', self.odom_callback, 10)
        self.ready_subscriber = self.create_subscription(Bool, '/amarsmer/controller_ready', self.ready_callback, 10)
        
        # Publishers
        self.thruster_input_sub = self.create_subscription(Float32MultiArray, "/thruster_input", self.thr_input_callback,10)
        self.controller_publisher = self.create_publisher(InCtrl, "/amarsmer/InCtrl", 10)
        self.pose_arrow_publisher = self.create_publisher(Marker, "/pose_arrow", 10)
        self.data_publisher = self.create_publisher(Float32MultiArray, "/monitoring_data", 10)

        # Create a client for path request
        self.client = self.create_client(RequestPath, '/path_request')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")
            
        self.future = None # Used for client requests

        # Initiate variables
        self.time_set = False
        self.initial_time = None
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.current_pose = None
        self.current_twist = None

        self.ready = False
        self.init = False

        self.u = None

        # Initialize controller 
        self.controller_path = Path()

        # Path computation parameters
        if self.controller_type == 'MPC':
            self.path_time = 2.5 # self.mpc_time 
            self.path_steps = 15 # self.mpc_horizon

        else:
            self.path_time = self.dt
            self.path_steps = 2

        # Initialize monitoring values
        self.monitoring = []

        column_names = ['t','x','y','psi','x_d','y_d','psi_d','u1','u2']

        if self.controller_type == 'AI':
            self.aiData_subscriber = self.create_subscription(Float32MultiArray, '/amarsmer/aiData', self.aiData_callback, 10)

            column_names.extend(['grad1', 'grad2', 'loss_x', 'loss_u'])
            self.AI_data = [0]*4

        self.monitoring.append(column_names)

        self.t_record = self.get_time()

        ctrl = self.controller_type # Inefficient but more readable
        date = datetime.today().strftime('%Y_%m_%d-%H_%M_%S')
        sim = 'simulation' if self.isSimulation else 'real'
        self.title = f'data/{ctrl}_data/{date}-{ctrl}_{sim}{network}{comment}_data'

    def get_time(self):
        s,ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    def odom_callback(self, msg: Odometry):
        pose, twist = cf.odometry(msg)

        self.current_pose = pose
        self.current_twist = twist

    def ready_callback(self, msg: Bool):
        self.ready = msg.data
        if msg.data:
            self.get_logger().info(f'Controller ready')

    def aiData_callback(self, msg):
        self.AI_data = msg.data

    def thr_input_callback(self, msg):
        self.u = msg.data

    def timer_callback(self):
        if not self.ready:
            return

        if not self.init:
            self.get_logger().info('Controller node initiated')
            self.init = True

        if not self.time_set:
            self.initial_time = self.get_time()
            self.time_set = True
        
        current_time = self.get_time() - self.initial_time

        ## Update path
        # Check if previous future is still pending
        if self.future is not None:
            if self.future.done():
                try:
                    result = self.future.result()
                    if result is not None:
                        self.controller_path = result.path
                    else:
                        self.get_logger().error("Service returned None.")
                except Exception as e:
                    self.get_logger().error(f"Service call raised exception: {e}")
                finally:
                    self.future = None
                return

        # Send new request
        request = RequestPath.Request()
        request.path_request.data = np.linspace(current_time, current_time + self.path_time, int(self.path_steps), dtype=float)

        self.future = self.client.call_async(request)

        if self.current_pose is None or self.current_twist is None:
            return
        
        current_state = [self.current_pose[0], # x
                         self.current_pose[1], # y
                         self.current_pose[5], # yaw
                         self.current_twist[0], # u
                         self.current_twist[1], # v
                         self.current_twist[5]] # r

        if self.controller_path.poses: # Make sure the path is not empty
            # Display the current desired pose if using gazebo
            if self.isSimulation:
                desired_pose = self.controller_path.poses[0].pose
                cf.create_pose_marker(desired_pose, self.pose_arrow_publisher) 

            target = cf.compute_target(self.controller_path, self.dt)

            # Send state and path to controller
            msg = InCtrl()
            msg.state = Float32MultiArray()
            msg.state.data = current_state
            msg.path = self.controller_path
            self.controller_publisher.publish(msg)

        if self.u is None:
            return

        # Update and save monitoring metrics to be graphed later
        if self.controller_path.poses:
            x_m = current_state[0]
            y_m = current_state[1]
            psi_m = current_state[2]

            x_d_m = target[0]
            y_d_m = target[1]
            psi_d_m = target[2]

            data_array = [current_time, x_m, y_m, psi_m, x_d_m, y_d_m, psi_d_m, self.u[0], self.u[1]]

            # # TODO
            if self.controller_type == 'AI':
                # data_array.extend(['grad1', 'grad2', 'loss_x', 'loss_u'])
                data_array.extend(self.AI_data)

            self.monitoring.append(data_array)

            publisher_msg = Float32MultiArray()
            publisher_msg.data = data_array
            self.data_publisher.publish(publisher_msg)

            if (current_time - self.t_record) > 0.1: # Update the saved file at set interval as doing so every step may corrupt the file if the callback is too frequent
                self.t_record = current_time
                np.save(self.title, self.monitoring)
        

rclpy.init()
node = Controller()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()