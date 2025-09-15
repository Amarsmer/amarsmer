#!/usr/bin/env python3

import time
import math
import torch
import numpy as np

def theta_s(x, y):
    return math.tanh(10.*x)*math.atan(1.*y)

class PyTorchOnlineTrainer:
    def __init__(self, robot, nn_model, monitor=None, Q=np.eye(6), R=np.eye(3)):

        self.robot = robot
        self.network = nn_model

        # Weighting matrices
        self.Q = Q
        self.R = R
        
        # État de l'apprentissage
        self.running = False
        self.training = False

        self.learning_rate = 5e-2
        self.optimizer_momentum = 0.1
        # Création de l'optimiseur (remplace partiellement la logique de backpropagate)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.optimizer_momentum)
        
        #TODO Add monitor support
        self.monitor = monitor
        self.last_gradient = [0, 0] if monitor else None

        # Variables init
        self.error = np.zeros(6)
        self.target = []
        self.command = []

        ## Modeling
        # Coefficients for gradient computation (removes unnecessary temp variable attribution)
        self.m = self.robot.mass
        self.Xudot = self.robot.added_masses[0]
        self.Nrdot = self.robot.added_masses[5]
        self.Iz = self.robot.inertia[-1]

        # Compute B matrix (constant)
        self.r = 0.15

        # NED not yet implemented, so the last row is reversed
        self.B = np.array([[1.        ,1.],
                      [0.        ,0.],
                     [self.r,-self.r]]) 

        self.command_set = False # Make sure inputs have been computed before recording data
        self.gradient_flag = None # Used to display gradient in terminal

    def computeNetworkInput(self, target):
        # Extract position and speed as column vectors
        position = np.array([self.robot.current_pose[x] for x in [0, 1, 5]]).reshape(-1, 1)
        speed = np.array([self.robot.current_twist[x] for x in [0, 1, 5]]).reshape(-1, 1)

        # Concatenate vertically (stack columns)
        state = np.vstack([position, speed])

        # Compute error as a column vector
        self.error = state - np.array(target).reshape(-1, 1)
        
        self.error[2] -= theta_s(state[0], state[1])

        weight_matrix = np.diag([0.1, 0.1, np.pi, 1, 1, 1])

        network_input = weight_matrix @ self.error
        
        """
        # Calculer l'entrée du réseau (erreur normalisée)
        network_input = np.zeros(6)
        network_input[0] = (self.error[0]) /10
        network_input[1] = (self.error[1]) /10
        network_input[2] = (self.error[2] - theta_s(state[0], state[1])) / np.pi
        network_input[3] = (self.error[3])
        network_input[4] = (self.error[4])
        network_input[5] = (self.error[5])
        """
        return network_input.ravel()

    def computeGradient(self, delta_t):
        grad_xk = np.array([[0.                              , 0.                            ],
                            [0.                              , 0.                            ],
                            [0.                              , 0.                            ],
                            [1/(self.m - self.Xudot)         , 1/(self.m - self.Xudot)       ],
                            [0.                              , 0.                            ],
                            [self.r/(self.Iz - self.Nrdot)   , -self.r/(self.Iz - self.Nrdot)]])
        
        grad_u = np.eye(2) # Not necessary but makes the code closer to the theoretical model

        w_tau = self.R @ self.tau
        u = np.linalg.pinv(self.B) @ w_tau

        gradxJ = 2 * (self.Q @ self.error)
        graduJ = 2 * (u)

        grad = delta_t * (grad_xk.transpose() @ gradxJ) + grad_u @ graduJ
        grad = grad.squeeze(-1)

        return grad

    def train(self, target):
        # Update monitor if available
        if self.monitor:
            self.monitor.set_target(target)
            
        network_input = self.computeNetworkInput(target)
        
        # Boucle d'apprentissage
        while self.running:
            # Get initial time for gradient computation later
            start_time = time.time()
            
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
            first_criteria = self.error.transpose() @ self.Q @ self.error + self.tau.transpose() @ self.R @ self.tau

            # Apply control input
            input_coefficient = 5
            # Appliquer les commandes au robot
            self.robot.move([self.command[0]*input_coefficient,self.command[1]*input_coefficient,0,0],
                      [0 for i in range(1,5)])
            
            # Wait before evaluating criteria again
            time.sleep(0.050)

            network_input = self.computeNetworkInput(target) # also updates error
           
            second_criteria = self.error.transpose() @ self.Q @ self.error + self.tau.transpose() @ self.R @ self.tau

            if self.training:
                delta_t = (time.time() - start_time)
                grad = self.computeGradient(delta_t)             

                # Update monitor with current gradient
                if self.monitor:
                    self.monitor.update(
                        position=position,
                        wheel_speeds=self.command,
                        gradient=grad,
                        cost=first_criteria
                    )
                
                # TODO: adapt learning strategy depending on criteria, currently does the same thing either way
                # Learning strategy
                if second_criteria <= first_criteria:
                    # Effectuer une étape d'apprentissage
                    self.optimizer.zero_grad()
                    
                    # Convertir le gradient
                    grad_tensor = torch.tensor(grad, dtype=torch.float32)
                    
                    # Effectuer une étape d'apprentissage personnalisée
                    self.manual_backward(input_tensor, grad_tensor, self.learning_rate, self.optimizer_momentum)
                else:
                    # Alternative si le critère ne s'améliore pas
                    # On peut soit ajouter du bruit soit quand même apprendre
                    self.optimizer.zero_grad()
                    
                    # Convertir le gradient
                    grad_tensor = torch.tensor(grad, dtype=torch.float32)
                    
                    # Effectuer une étape d'apprentissage personnalisée
                    self.manual_backward(input_tensor, grad_tensor, self.learning_rate, self.optimizer_momentum)
        
        # Stop the robot after learning
        self.robot.move([0,0,0,0],
                      [0 for i in range(1,5)])
        self.running = False
    
    def manual_backward(self, inputs, grad_tensor, learning_rate, momentum):
        """
        Effectue manuellement une étape de rétropropagation avec le gradient fourni
        
        Args:
            inputs: les entrées du réseau
            grad_tensor: le gradient du critère par rapport aux sorties
            learning_rate: le taux d'apprentissage
            momentum: le facteur de momentum
        """
        # Réinitialiser les gradients
        self.optimizer.zero_grad()
        
        # Forward pass pour établir le graphe de calcul
        outputs = self.network(inputs)
        
        # Rétropropager le gradient directement
        # C'est ici que nous connectons notre gradient externe au graphe PyTorch
        outputs.backward(gradient=grad_tensor)
        
        # Mise à jour des poids
        for param in self.network.parameters():
            if param.grad is not None:
                # Mise à jour manuelle avec le momentum si nécessaire
                param.data.add_(param.grad, alpha=-learning_rate)