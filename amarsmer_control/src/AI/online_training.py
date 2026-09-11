#!/usr/bin/env python3

import time
import math
import torch
import numpy as np
import sympy as sp
import custom_functions as cf

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def inRobotFrame(robot_coords, target_coords):
    x_r,y_r,psi_r,_,_,_ = robot_coords
    x_t,y_t,cpsi_t,spsi_t,_,_,_ = target_coords

    cpsi_r = np.cos(psi_r)
    spsi_r = np.sin(psi_r)

    psi_r = np.arctan2(spsi_r,cpsi_r)
    psi_t = np.arctan2(spsi_t,cpsi_t)

    x = (x_t - x_r)*cpsi_r + (y_t - y_r)*spsi_r
    y = (y_t - y_r)*cpsi_r - (x_t - x_r)*spsi_r
    # psi = wrap_angle(psi_t) - wrap_angle(psi_r)
    cpsi = cpsi_t - cpsi_r
    spsi = spsi_t - spsi_r

    return x[0],y[0],cpsi[0],spsi[0]

class PyTorchOnlineTrainer:
    def __init__(self, nn_model, in_learning_rate = 5e-4, in_Q=np.eye(6), in_R=np.eye(3), nb_thr = 2):

        self.network = nn_model
        self.unwrap = False

        self.nb_thr = nb_thr

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

        self.robot_frame = [0,0,0,0]

        ## Modeling
        # B matrix, NED not yet implemented, so the last row is reversed
        R = sp.Symbol('R')
        hl = sp.Symbol('hl')

        if self.nb_thr == 2:
            B = sp.Matrix([[1, 1],
                           [0, 0],
                           [R,-R]])

        elif self.nb_thr == 3:
            B = sp.Matrix([[1, 1, 0],
                           [0, 0, 1],
                           [R,-R, hl]])

        # Coefficients for gradient computation (reduces unnecessary temp variable attribution in loop since it's constant)
        # Read YAML file for robot's properties
        mass, inertia, added_masses, viscous_drag, _ = cf.read_model()

        radius = 0.15
        hl = 0.3
        planar_added_mass = [added_masses[i] for i in [0,1,5]]
        planar_dampening = [viscous_drag[i] for i in [0,1,5]]
        
        self.compute_gradient,_,_,_ = cf.build_grad(B, mass, planar_added_mass, inertia[-1], planar_dampening, radius, hl, in_Q, in_R)

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
        self.target = np.array(in_target).reshape(-1, 1)

    def updateState(self, in_state):
        self.state = in_state

    def computeError(self):

        def theta_s(x, y): # Angle skew, used to prevent the singularity in x=0
            return math.tanh(5.*x)*math.atan(10.*y)

        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        # Compute error as a column vector
        self.state_display = self.state.copy()
        self.target_display = self.target.copy()

        self.robot_frame = inRobotFrame(self.state, self.target)
        state = self.state.copy().ravel()
        target = self.target.copy().ravel()
        dx = state[0] - target[0]
        dy = state[1] - target[1]
        skew = np.arctan2(-dy, -dx)
        # skew = float(np.arctan(state[1]/state[0]))

        # skew = -float(np.arctan((target[1]-state[1])/(target[0]-state[0])))
        self.skew = skew

        if self.nb_thr == 2 :
            skew_target = skew

        elif self.nb_thr == 3 :
            skew_target = np.arctan2(target[3],target[2])

        # error = self.state - self.target
        error = np.array([state[0] - target[0],
                          state[1] - target[1],
                          np.cos(state[2]) - np.cos(skew_target),
                          np.sin(state[2]) - np.sin(skew_target),
                          state[3] - target[3],
                          state[4] - target[4],
                          state[5] - target[5],
                          ]).reshape(-1, 1)

        ### Apply skew angle ###
        # angle = self.target[2]
        # heading = np.arctan2(error[1],error[0])
        # d = np.hypot(error[1],error[0])

        # alpha = 10
        # beta = 0.5
        # ea = np.exp(-d**2*alpha)
        # eb = 1-np.exp(-d**2*beta)

        # skewed_angle = ea*angle + eb*heading
        # skewed_angle = angle

        return error, skew_target

    def computeNetworkInput(self, error):
        # Weight matrix used for input normalization
        # weight_matrix = np.diag([1/10, 1/10, 1/(2*np.pi), 1, 1, 1/(2*np.pi)])
        weight_matrix = np.diag([1/5, 1/5, 1, 1, 1, 1, 1/(2*np.pi)])
        network_input = weight_matrix @ error
        
        return network_input.ravel()

    def train(self, target):
        # Training loop
        while self.running:
            while self.state is None:
                time.sleep(0.001)

            # Get initial time for gradient computation later
            start_time = time.time()

            error, skew_angle = self.computeError()
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
                state = self.state.copy()
                skew_target = self.target.copy()
                skew_target[2] = np.cos(skew_angle)
                skew_target[3] = np.sin(skew_angle)
                grad = self.compute_gradient(state, skew_target, self.u, delta_t, 1, 1000).squeeze()
                
                # Convert to tensor grad
                grad_tensor = torch.tensor(grad, dtype=torch.float32)

                self.gradient_display = grad.copy()

                # Backprop using external gradient
                self.optimizer.zero_grad()
                u_tensor.backward(gradient=grad_tensor)
                self.optimizer.step()

            else:
                self.gradient_display = np.array([0,0])

            self.previous_target = self.target
            self.previous_state = self.state

            # Monitoring data for debugging purposes
            self.state_train_display = self.state.copy()
            self.error_display = error.copy()
            self.input_display = network_input.copy()

            if not self.trainer_set: # Used for data recording purposes
                self.trainer_set = True