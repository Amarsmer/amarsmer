#!/usr/bin/env python3

# rclpy
from rclpy.node import Node
import rclpy

# Common python libraries
import numpy as np
from functools import partial

# ROS2 msg libraries
from std_msgs.msg import Bool, Float64, Float64MultiArray

# Custom libraries
from amarsmer_control import ROV
import custom_functions as cf

class Input(Node):
    def __init__(self):

        super().__init__('input_comp', namespace='amarsmer')

        self.declare_parameter('nb_thr', 2) 
        self.nb_thrusters = self.get_parameter('nb_thr').get_parameter_value().integer_value

        self.input_publisher = self.create_publisher(Float64MultiArray, "/thruster_input", 10)

        self._subscriptions = []

        max_topics = self.nb_thrusters

        for i in range(1, max_topics+1):
            topic = f"/amarsmer/cmd_thruster{i}"

            sub = self.create_subscription(
                Float64,
                topic,
                partial(self.u_callback, thr_id=i),
                10,
            )

            topic = f"/amarsmer/cmd_thruster{i}_steering"

            sub = self.create_subscription(
                Float64,
                topic,
                partial(self.beta_callback, thr_id=i),
                10,
            )

            self.get_logger().info(f'Thruster {i} subscription done')
            self._subscriptions.append(sub)

        self.rov = ROV(self, thrust_visual = True)

        self.timer = self.create_timer(0.01, self.run)
        self.thr_input = [0]*self.nb_thrusters
        self.angle_input = [0]*self.nb_thrusters

        self.get_logger().info("End of __init__")

    def u_callback(self, msg, thr_id):
        # self.get_logger().info(f"Received from sensor {thr_id}")
        self.thr_input[thr_id-1] = msg.data

    def beta_callback(self, msg, thr_id):
        # self.get_logger().info(f"Received from sensor {thr_id}")
        self.angle_input[thr_id-1] = msg.data

    def dyn(self, u):
        n = np.sign(u)*np.sqrt(abs(u))*100
        return n

    def run(self):
        if not self.rov.ready():
            self.get_logger().info('Not ready')
            return

        # NWU to NED conversion matrix
        R = -np.eye(3)
        R[0,0] = 1
        Cn = np.block([[R, np.zeros((3,3))],
                       [np.zeros((3,3)), R]])

        # Plasmar tau
        B_p = np.array(self.rov.TAM(*self.angle_input), dtype=np.float64)

        # self.get_logger().info(f'B_p : \n{B_p}')

        K_p = 40*np.eye(4)

        u = np.array(self.thr_input).reshape(-1,1)
        norm_p = 1/40*np.eye(4)
        u_p = norm_p @ u

        tau_p = Cn @ B_p @ K_p @ u_p

        # Output tau
        sq2 = 1/np.sqrt(2)

        B_o = np.array([[sq2     ,  sq2     ,  sq2     ,  sq2     ,  0   ,  0   , 0   ,  0   ],
                        [sq2     , -sq2     ,  sq2     , -sq2     ,  0   ,  0   , 0   ,  0   ],
                        [0       ,  0       ,  0       ,  0       ,  1   ,  1   , 1   ,  1   ],
                        [0       ,  0       ,  0       ,  0       , -0.23,  0.23, 0.23, -0.23],
                        [0       ,  0       ,  0       ,  0       , -0.22, -0.22, 0.22,  0.22],
                        [0.54*sq2, -0.54*sq2, -0.54*sq2,  0.54*sq2,  0   ,  0   , 0   ,  0   ]])

        K_o = 60*np.sqrt(2)*np.eye(8)

        u_o = np.linalg.inv(K_o) @ np.linalg.pinv(B_o) @ tau_p

        u_t = self.dyn(u_o)

        # self.get_logger().info(f'Uo = {u_o}')

        # Publish 

        publisher_msg = Float64MultiArray()
        publisher_msg.data = u_t
        self.input_publisher.publish(publisher_msg)

rclpy.init()
node = Input()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()