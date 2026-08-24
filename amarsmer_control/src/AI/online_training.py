#!/usr/bin/env python3

import time
import math
import torch
import numpy as np
import sympy as sp
import custom_functions as cf

def theta_s(x, y): # Angle skew, used to prevent the singularity in x=0
    return math.tanh(5.*x)*math.atan(10.*y)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def inRobotFrame(robot_coords, target_coords):
    x_r,y_r,psi_r,_,_,_ = robot_coords
    x_t,y_t,psi_t,_,_,_ = target_coords

    cos = np.cos
    sin = np.sin

    x = (x_t - x_r)*cos(psi_r) + (y_t - y_r)*sin(psi_r)
    y = (y_t - y_r)*cos(psi_r) - (x_t - x_r)*sin(psi_r)
    psi = wrap_angle(psi_t) - wrap_angle(psi_r)

    return x[0],y[0],psi[0]

class PyTorchOnlineTrainer:
    def __init__(self, nn_model, in_learning_rate = 5e-4, in_Q=np.eye(6), in_R=np.eye(3)):

        self.network = nn_model
        self.unwrap = False

        # Weighting matrices
        self.Q = in_Q
        self.R = in_R
        
        # Training state
        self.running = False
        self.training = True

        self.learning_rate = in_learning_rate

        # Set up optimizer (partially replaces backpropagation)
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=self.learning_rate)

        # Variables init
        self.state = None
        self.target = None
        self.error = None
        self.u = np.zeros(2)
        self.loss = None

        self.previous_state = None
        self.previous_target = None

        self.robot_frame = [0,0,0]

        ## Modeling
        # B matrix, NED not yet implemented, so the last row is reversed
        R = sp.Symbol('R')

        B = sp.Matrix([[1, 1],
                       [0, 0],
                       [R,-R]])

        # Coefficients for gradient computation (reduces unnecessary temp variable attribution in loop since it's constant)
        # Read YAML file for robot's properties
        mass, inertia, added_masses, viscous_drag, _ = cf.read_model()

        radius = 0.15
        planar_added_mass = [added_masses[i] for i in [0,1,5]]
        planar_dampening = [viscous_drag[i] for i in [0,1,5]]
        
        self.compute_gradient,_,_,_ = cf.build_grad(B, mass, planar_added_mass, inertia[-1], planar_dampening, radius, in_Q, in_R)

        self.trainer_set = False # Make sure inputs have been computed before recording data

        # Monitoring variables, meant to be displayed in terminal
        self.gradient_display = None
        self.input_display = None 
        self.error_display = None 
        self.delta_t_display = None
        self.skew = None
        self.state_display = None
        self.loss_display = np.zeros(2)

        self.target_display = None

    def updateTarget(self, in_target):
        temp_target = in_target

        if self.unwrap and self.previous_target is not None:
            temp_target[2] = np.unwrap([self.previous_target[2],temp_target[2]])[-1]

        self.target = np.array(temp_target).reshape(-1, 1)

    def updateState(self, in_state):
        temp_state = in_state

        if self.unwrap and self.previous_state is not None:
            temp_state[2] = np.unwrap([self.previous_state[2],temp_state[2]])[-1]

        self.state = temp_state

    def computeError(self):
        # Compute error as a column vector
        self.state_display = self.state.copy()
        self.target_display = self.target.copy()

        error = self.state - self.target

        self.robot_frame = inRobotFrame(self.state, self.target)
        error[:3] = np.array(self.robot_frame).reshape(-1, 1)

        # skew = theta_s(self.state[0], self.state[1])
        angle = error[2]
        skew = np.arctan(error[1],error[0])
        d = np.hypot(error[1],error[0])
        e = np.exp(-2*d)
        # error[2] -= skew # Yaw skew
        error[2] = e * angle + (1-e) * skew

        # Apply angle disambiguation
        # error[2] = 2*np.sin(wrap_angle(error[2])/2)
        
        # self.skew = skew # Monitoring
        
        return error

    def computeNetworkInput(self, error):
        # Weight matrix used for input normalization
        weight_matrix = np.diag([1/10, 1/10, 1/(2*np.pi), 1, 1, 1/(2*np.pi)])
        network_input = weight_matrix @ error
        
        return network_input.ravel()

    def train(self, target):
        # Training loop
        while self.running:
            while self.state is None:
                time.sleep(0.001)

            # Get initial time for gradient computation later
            start_time = time.time()

            error = self.computeError()
            self.error_display = error.copy()
            network_input = self.computeNetworkInput(error)
            
            # Prepare input
            input_tensor = torch.tensor(network_input, dtype=torch.float32, requires_grad=self.training)

            # Forward pass
            if self.training:
                u_tensor = self.network(input_tensor)
            else:
                with torch.no_grad():
                    u_tensor = self.network(input_tensor)

            # Scale output
            input_coefficient = 40.0
            u_tensor = input_coefficient * u_tensor

            # Apply control input (convert ONLY for the robot)
            self.u = u_tensor.detach().cpu().numpy().copy().reshape(-1, 1)

            # Compute loss, both for monitoring and later for backpropagation
            crit_x = error.transpose() @ self.Q @ error
            crit_u = self.u.transpose() @ self.R @ self.u
            self.loss = crit_x + crit_u

            self.loss_display = np.array([crit_x, crit_u])

            ### Training step
            if self.training:
                delta_t = (time.time() - start_time)
                self.delta_t_display = delta_t

                # Manual gradient computation
                grad = self.compute_gradient(self.state, error, self.u, delta_t, 1, 1000).squeeze()
                
                # Convert to tensor grad
                grad_tensor = torch.tensor(grad, dtype=torch.float32)

                self.gradient_display = grad.copy()

                # Backprop using external gradient
                self.optimizer.zero_grad()
                u_tensor.backward(gradient=grad_tensor)
                self.optimizer.step()

            self.previous_target = self.target
            self.previous_state = self.state

            # Monitoring data for debugging purposes
            self.state_train_display = self.state.copy()
            self.error_display = error.copy()
            self.input_display = network_input.copy()

            if not self.trainer_set: # Used for data recording purposes
                self.trainer_set = True