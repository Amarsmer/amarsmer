#!/usr/bin/env python3

import time
import math
import torch
import numpy as np

def theta_s(x, y):
    return math.tanh(10.*x)*math.atan(1.*y)

class PyTorchOnlineTrainer:
    def __init__(self, robot, nn_model, in_learning_rate = 5e-4, in_momentum = 0.2, Q=np.eye(6), R=np.eye(3)):

        self.robot = robot
        self.network = nn_model

        # Weighting matrices
        self.Q = Q
        self.R = R
        
        # Training state
        self.running = False
        self.training = False

        self.learning_rate = in_learning_rate
        self.optimizer_momentum = in_momentum

        # Set up optimizer (partially replaces backpropagation)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.optimizer_momentum)

        # Variables init
        self.state = None
        self.error = None
        self.target = None
        self.command = []
        self.state_display = None

        ## Modeling
        # Compute B matrix (constant)
        self.r = 0.15

        # NED not yet implemented, so the last row is reversed
        self.B = np.array([[1.        ,1.],
                      [0.        ,0.],
                     [self.r,-self.r]]) 

        # Coefficients for gradient computation (removes unnecessary temp variable attribution in loop)
        self.m = self.robot.mass
        self.Xudot = self.robot.added_masses[0]
        self.Nrdot = self.robot.added_masses[5]
        self.Iz = self.robot.inertia[-1]
        
        self.grad_xk = np.array([[0.                              , 0.                            ],
                                 [0.                              , 0.                            ],
                                 [0.                              , 0.                            ],
                                 [1/(self.m - self.Xudot)         , 1/(self.m - self.Xudot)       ],
                                 [0.                              , 0.                            ],
                                 [self.r/(self.Iz - self.Nrdot)   , -self.r/(self.Iz - self.Nrdot)]])

        self.command_set = False # Make sure inputs have been computed before recording data
        self.gradient_display = None # Used to display gradient in terminal
        self.input_display = None # Used to display network input in terminal
        self.error_display = None # Used to display error in terminal

    def updateTarget(self, in_target):
        self.target = in_target

    def updateState(self, in_state):
        self.state = in_state

    def computeError(self):
        # Compute error as a column vector
        error = self.state - np.array(self.target).reshape(-1, 1)
        # self.error[2] -= theta_s(state[0], state[1]) # Yaw skew

        return error

    def computeNetworkInput(self, error):
        # Weight matrix used for input normalization
        weight_matrix = np.diag([1/4, 1/4, 1/np.pi, 1, 1, 1])
        network_input = weight_matrix @ error
        
        return network_input.ravel()

    def computeGradient(self, delta_t, error):
        w_tau = self.R @ self.tau
        u = np.linalg.pinv(self.B) @ w_tau

        gradxJ = 2 * (self.Q @ error)
        graduJ = 2 * (u)

        grad = delta_t * (self.grad_xk.transpose() @ gradxJ) + graduJ
        grad = grad.squeeze(-1) # Removes the dimensions of size 1

        return grad

    def train(self, target):
        # Training loop
        while self.running:
            # Get initial time for gradient computation later
            start_time = time.time()

            error = self.computeError()
            network_input = self.computeNetworkInput(error)

            # Monitoring data for debugging purposes
            self.state_train_display = self.state
            self.error_display = self.error
            self.input_display = network_input
            
            # Forward pass - get vector input
            input_tensor = torch.tensor(network_input, dtype=torch.float32)
            if self.training:
                # Use gradient when training
                input_tensor.requires_grad_(True)
                self.command = self.network(input_tensor).tolist()
            else:
                # No gradients otherwise
                with torch.no_grad():
                    self.command = self.network(input_tensor).tolist()
            
            if not self.command_set: # Used for data recording purposes
                self.command_set = True
            
            # Evaluate criteria before moving the robot
            self.command = np.array(self.command).reshape(-1, 1)
            self.tau = self.B @ self.command

            first_criteria = error.transpose() @ self.Q @ error + self.tau.transpose() @ self.R @ self.tau

            # Apply control input
            input_coefficient = 40 # The network outputs 1 at max and the thrusters are expected to be able to output 40 newtons
            self.robot.move([self.command[0]*input_coefficient,self.command[1]*input_coefficient,0,0],
                      [0 for i in range(1,5)])
            
            # Wait before evaluating criteria again
            time.sleep(0.050)

            # Update error
            error = self.computeError()
           
            second_criteria = error.transpose() @ self.Q @ error + self.tau.transpose() @ self.R @ self.tau

            if self.training:
                delta_t = (time.time() - start_time)
                grad = self.computeGradient(delta_t, error)             
                
                # TODO: adapt learning strategy depending on criteria, currently does the same thing either way
                # Learning strategy
                if second_criteria <= first_criteria:
                    # Do a learning step
                    self.optimizer.zero_grad()
                    
                    # Convert gradient
                    grad_tensor = torch.tensor(grad, dtype=torch.float32)
                    
                    # Do a custom learning step
                    self.manual_backward(input_tensor, grad_tensor, self.learning_rate, self.optimizer_momentum)
                else:
                    # If the criteria does not improve
                    # Either add noise or learn regardless
                    self.optimizer.zero_grad()
                    
                    # Convert gradient
                    grad_tensor = torch.tensor(grad, dtype=torch.float32)
                    
                    # Do a custom learning step
                    self.manual_backward(input_tensor, grad_tensor, self.learning_rate, self.optimizer_momentum)
        
        # Stop the robot after learning
        self.robot.move([0,0,0,0],
                      [0 for i in range(1,5)])
        self.running = False
    
    def manual_backward(self, inputs, grad_tensor, learning_rate, momentum):
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.network(inputs)
        
        # Manually backpropagate
        # Connect externally computed gradient to pytorch
        outputs.backward(gradient=grad_tensor)
        
        # Update weights
        for param in self.network.parameters():
            if param.grad is not None:
                param.data.add_(param.grad, alpha=-learning_rate)