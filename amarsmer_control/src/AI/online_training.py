#!/usr/bin/env python3

import time
import math
import torch
import numpy as np

def theta_s(x, y): # Angle skew, used to prevent the singularity in x=0
    return math.tanh(5.*x)*math.atan(10.*y)

class PyTorchOnlineTrainer:
    def __init__(self, robot, nn_model, in_learning_rate = 5e-4, in_momentum = 0.0, Q=np.eye(6), R=np.eye(3)):

        self.robot = robot
        self.network = nn_model

        # Weighting matrices
        self.Q = Q
        self.R = R
        
        # Training state
        self.running = False
        self.training = True

        self.learning_rate = in_learning_rate
        self.optimizer_momentum = in_momentum

        # Set up optimizer (partially replaces backpropagation)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.optimizer_momentum)

        # Variables init
        self.state = None
        self.error = None
        self.target = None
        self.u = np.zeros(2)
        self.loss = None

        ## Modeling
        # Compute B matrix (constant)
        self.radius = 0.15 # radius of the robot

        # NED not yet implemented, so the last row is reversed
        self.B = np.array([[1.        ,1.],
                           [0.        ,0.],
                           [self.radius,-self.radius]]) 

        # Coefficients for gradient computation (removes unnecessary temp variable attribution in loop)
        self.m = self.robot.mass

        self.Xudot = self.robot.added_masses[0]
        self.Yvdot = self.robot.added_masses[1]
        self.Nrdot = self.robot.added_masses[5]

        self.Xu = self.robot.viscous_drag[0]
        self.Yv = self.robot.viscous_drag[1]
        self.Nr = self.robot.viscous_drag[5]

        self.Iz = self.robot.inertia[-1]
        
        self.grad_xdk = np.array([[0.                              , 0.                            ],
                                  [0.                              , 0.                            ],
                                  [0.                              , 0.                            ],
                                  [1/(self.m - self.Xudot)         , 1/(self.m - self.Xudot)       ],
                                  [0.                              , 0.                            ],
                                  [self.radius/(self.Iz - self.Nrdot)   , -self.radius/(self.Iz - self.Nrdot)]])

        self.command_set = False # Make sure inputs have been computed before recording data

        # Monitoring variables, meant to be displayed in terminal
        self.gradient_display = None
        self.input_display = None 
        self.error_display = None 
        self.delta_t_display = None
        self.skew = None
        self.state_display = None
        self.loss_display = np.zeros(2)

    def updateTarget(self, in_target):
        self.target = in_target

    def updateState(self, in_state):
        self.state = in_state

    def computeError(self):
        # Compute error as a column vector
        error = self.state - np.array(self.target).reshape(-1, 1)
        skew = theta_s(self.state[0], self.state[1])
        error[2] -= skew # Yaw skew

        self.skew = skew # Monitoring
        
        return error

    def computeNetworkInput(self, error):
        # Weight matrix used for input normalization
        weight_matrix = np.diag([1/10, 1/10, 1/np.pi, 1, 1, 1])
        network_input = weight_matrix @ error
        
        return network_input.ravel()

    def computeGradient(self, delta_t, error, alpha1 = 0, alpha2 = 1000):
        x,y,psi,u,v,r = self.state.ravel()

        gradxJ = 2 * (self.Q @ error)
        graduJ = 2 * (self.R @ self.u)

        fod_grad = self.grad_xdk # first order derivative gradient

        # second order derivative gradient, done in multiple steps to increase readability and ease of debugging
        cos = np.cos
        sin = np.sin

        fracmXu = self.m + self.Xudot
        fracmYv = self.m + self.Yvdot
        fracIzNr = self.Iz + self.Nrdot

        grad3_A = self.radius*v*(self.m - self.Yvdot)/fracIzNr # each "A" element is dependent on the radius and may need a change of sign when NED is implemented
        grad3_B = self.Xu/fracmXu

        grad4_A = u*r/fracIzNr
        grad4_B = r/fracmXu

        grad5_A = (self.radius*self.Nr)/fracIzNr
        grad5_B = v*(self.Yvdot-self.Xudot)/fracmXu

        sod_grad = np.array([[cos(psi)/fracmXu                                 , cos(psi)/fracmXu                                  ],
                            [sin(psi)/fracmXu                                  , sin(psi)/fracmXu                                  ],
                            [self.radius/fracIzNr                              , -self.radius/fracIzNr                             ],
                            [(-grad3_A + grad3_B)/fracmXu                      , (grad3_A + grad3_B)/fracmXu                       ],
                            [(-grad4_A + grad4_B)*(self.Xudot - self.m)/fracmYv, (grad4_A + grad4_B)*(self.Xudot - self.m)/fracmYv ],
                            [(-grad5_A + grad5_B)/fracIzNr                     , (grad5_A + grad5_B)/fracIzNr                      ]]) 

        # cost function gradient
        time_gradient = alpha1 * delta_t * fod_grad + alpha2 * 0.5*delta_t**2 * sod_grad

        grad = (time_gradient.transpose() @ gradxJ) + graduJ
        grad = grad.squeeze(-1) # Removes the dimensions of size 1

        return grad

    def train(self, target):
        # Training loop
        while self.running:
            # Get initial time for gradient computation later
            start_time = time.time()

            error = self.computeError()
            network_input = self.computeNetworkInput(error)
            
            # Forward pass - get vector input
            input_tensor = torch.tensor(network_input, dtype=torch.float32)
            if self.training:
                # Use gradient when training
                input_tensor.requires_grad_(True)
                self.u = self.network(input_tensor).tolist()
            else:
                # No gradients otherwise
                with torch.no_grad():
                    self.u = self.network(input_tensor).tolist()
            
            if not self.command_set: # Used for data recording purposes
                self.command_set = True
            
            input_coefficient = 40 # The network outputs 1 at max and the thrusters are expected to be able to output 40 newtons
            self.u = input_coefficient * np.array(self.u).reshape(-1, 1)

            # Apply control input
            self.robot.move([self.u[0],self.u[1],0,0],
                      [0 for i in range(1,5)])

            # Compute loss, both for monitoring and later for backpropagation
            crit_x = error.transpose() @ self.Q @ error
            crit_u = self.u.transpose() @ self.R @ self.u
            self.loss = crit_x + crit_u

            self.loss_display = np.array([crit_x, crit_u])

            if self.training:
                delta_t = (time.time() - start_time)

                self.delta_t_display = delta_t

                grad = self.computeGradient(delta_t, error)

                self.gradient_display = grad     
                
                # TODO: adapt learning strategy depending on loss, currently does the same thing either way
                # Learning strategy
                # Do a learning step
                self.optimizer.zero_grad()
                
                # Convert gradient
                grad_tensor = torch.tensor(grad, dtype=torch.float32)
                
                # Do a custom learning step
                self.manual_backward(input_tensor, grad_tensor, self.learning_rate, self.optimizer_momentum)

            # Monitoring data for debugging purposes
            self.state_train_display = self.state
            self.error_display = error
            self.input_display = network_input

        # Stop the robot after learning
        self.robot.move([0,0,0,0],
                      [0 for i in range(1,5)])
        # self.running = False
    
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