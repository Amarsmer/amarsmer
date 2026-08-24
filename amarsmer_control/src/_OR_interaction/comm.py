#!/usr/bin/env python3

# rclpy
from rclpy.node import Node
import rclpy

# Common python libraries
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import socket
import sys

# ROS2 msg libraries
from std_msgs.msg import Float64MultiArray

# Custom libraries
from amarsmer_control import ROV
import custom_functions as cf

class Comm(Node):
    def __init__(self):

        super().__init__('OR_communication', namespace='amarsmer')

        self.declare_parameter('IP', '127.0.0.1')
        self.declare_parameter('Port', 61022)

        self.thruster_input_sub = self.create_subscription(Float64MultiArray, "/thruster_input", self.thr_input_callback,10)

        self.nb_thrusters = 8
        self.thr_input = [0]*self.nb_thrusters

        self.timer = self.create_timer(0.01, self.send)

    def thr_input_callback(self, msg: Float64MultiArray):
        self.thr_input = msg.data

    def calculate_checksum(self, nmea_str):
        """Calculate NMEA checksum."""
        checksum = 0
        for char in nmea_str:
            checksum ^= ord(char)
        return f"{checksum:02X}"

    def send(self):
        # Get the values from the sliders
        HT_nd1_L0 = self.thr_input[0]
        HT_nd2_L0 = self.thr_input[1]
        HT_nd3_L0 = self.thr_input[2]
        HT_nd4_L0 = self.thr_input[3]

        HT_nd1_L1 = 0.0
        HT_nd2_L1 = 0.0
        HT_nd3_L1 = 0.0
        HT_nd4_L1 = 0.0

        VT_nd1 = self.thr_input[4]
        VT_nd2 = self.thr_input[5]
        VT_nd3 = self.thr_input[6]
        VT_nd4 = self.thr_input[7]

        # Create the NMEA string
        nmea_data = f"CI_IN_TCS_SET,{HT_nd1_L0:.2f},{HT_nd2_L0:.2f},{HT_nd3_L0:.2f},{HT_nd4_L0:.2f},{HT_nd1_L1:.2f},{HT_nd2_L1:.2f},{HT_nd3_L1:.2f},{HT_nd4_L1:.2f},{VT_nd1:.2f},{VT_nd2:.2f},{VT_nd3:.2f},{VT_nd4:.2f}"
        checksum = self.calculate_checksum(nmea_data)
        message = f"${nmea_data}*{checksum}\r\n"

        # Find ip and port from the UI
        # ip = self.DestinationIP.text()
        # port = self.DestinationPort.value()

        ip = self.get_parameter('IP').get_parameter_value().string_value
        port = self.get_parameter('Port').get_parameter_value()._integer_value
        
        # Send the NMEA string to the specified IP and port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(message.encode(), (ip, port))
            # self.label.setText(message)
        except Exception as e:
        #    self.label.setText("Failed to send: {e}")
           self.get_logger().error(f"Failed to send: {e}")

rclpy.init()
node = Comm()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()