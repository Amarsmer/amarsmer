#!/usr/bin/env python3

import time
import math
import torch
import numpy as np

def theta_s(x, y):
    return math.tanh(10.*x)*math.atan(1.*y)

class PyTorchOnlineTrainer:
    def __init__(self, robot, nn_model, monitor=None, Q=np.eye(6), R=np.eye(3)):
        """
        Args:
            robot (Robot): instance du robot suivant le modèle de ZMQPioneerSimulation
            nn_model (PioneerNN): modèle PyTorch du réseau de neurones
        """
        self.robot = robot
        self.network = nn_model

        # Facteurs de normalisation 
        self.alpha = [1/6, 1/6, 1/(math.pi)]
        self.Q = Q
        self.R = R
        
        # État de l'apprentissage
        self.running = False
        self.training = False

        # Création de l'optimiseur (remplace partiellement la logique de backpropagate)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.72, momentum=0)
        
        # Add monitor support
        self.monitor = monitor
        self.last_gradient = [0, 0] if monitor else None

        self.command = []

        self.target = []

        self.r = 0.15

        self.B = np.array([[1.        ,1.],
                      [0.        ,0.],
                     [self.r,-self.r]]) 

        self.command_set = False
        self.gradient_flag = None
    def train(self, target):
        # Update monitor if available
        if self.monitor:
            self.monitor.set_target(self.target)
            
        # Extract position and speed as column vectors
        position = np.array([self.robot.current_pose[x] for x in [0, 1, 5]]).reshape(-1, 1)
        speed = np.array([self.robot.current_twist[x] for x in [0, 1, 5]]).reshape(-1, 1)

        # Concatenate vertically (stack columns)
        state = np.vstack([position, speed])

        # Compute error as a column vector
        error = state - np.array(target).reshape(-1, 1)
        
        # Calculer l'entrée du réseau (erreur normalisée)
        network_input = np.zeros(6)
        network_input[0] = (error[0])
        network_input[1] = (error[1])
        network_input[2] = (error[2] - theta_s(state[0], state[1]))
        network_input[3] = (error[3])
        network_input[4] = (error[4])
        network_input[5] = (error[5])
        
        # Boucle d'apprentissage
        while self.running:
            # Mesurer le temps de début pour le calcul du delta_t
            debut = time.time()
            
            # Forward pass - obtenir les commandes de vitesse des roues
            input_tensor = torch.tensor(network_input, dtype=torch.float32)
            if self.training:
                # En mode apprentissage, nous voulons les gradients
                input_tensor.requires_grad_(True)
                self.command = self.network(input_tensor).tolist()
            else:
                # En mode évaluation, pas besoin de gradients
                with torch.no_grad():
                    self.command = self.network(input_tensor).tolist()
            
            if not self.command_set:
                self.command_set = True
            
            # Calculer le critère avant de déplacer le robot
            """
            alpha_x = self.alpha[0]
            alpha_y = self.alpha[1]
            alpha_teta = self.alpha[2]
            
            crit_av = (alpha_x * alpha_x * (position[0] - self.target[0])**2 + 
                       alpha_y * alpha_y * (position[1] - self.target[1])**2 + 
                       alpha_teta * alpha_teta * (position[2] - self.target[2] - 
                                                 theta_s(position[0], position[1]))**2)
            """
            self.command = np.array(self.command).reshape(-1, 1)
            tau = self.B @ self.command
            crit_av = error.transpose() @ self.Q @ error + tau.transpose() @ self.R @ tau
            coeff = 10
            # Appliquer les commandes au robot
            self.robot.move([self.command[1]*coeff,self.command[0]*coeff,0,0],
                      [0 for i in range(1,5)])
            
            # Attendre un court instant
            time.sleep(0.050)

            # Extract position and speed as column vectors
            position = np.array([self.robot.current_pose[x] for x in [0, 1, 5]]).reshape(-1, 1)
            speed = np.array([self.robot.current_twist[x] for x in [0, 1, 5]]).reshape(-1, 1)

            # Concatenate vertically (stack columns)
            state = np.vstack([position, speed])

            # Compute error as a column vector
            error = state - np.array(target).reshape(-1, 1)
            
            # Mettre à jour l'entrée du réseau
            network_input[0] = (error[0])
            network_input[1] = (error[1])
            network_input[2] = (error[2] - theta_s(state[0], state[1]))
            network_input[3] = (error[3])
            network_input[4] = (error[4])
            network_input[5] = (error[5])
            
            """
            # Calculer le critère après déplacement
            crit_ap = (alpha_x * alpha_x * (position[0] - self.target[0])**2 + 
                      alpha_y * alpha_y * (position[1] - self.target[1])**2 + 
                      alpha_teta * alpha_teta * (position[2] - self.target[2] - 
                                                theta_s(position[0], position[1]))**2)
            """
            crit_ap = error.transpose() @ self.Q @ error + tau.transpose() @ self.R @ tau
            # Apprentissage (si activé)
            if self.training:
                delta_t = (time.time() - debut)
                m = self.robot.mass
                Xudot = self.robot.added_masses[0]
                Nrdot = self.robot.added_masses[5]
                Iz = self.robot.inertia[-1]

                grad_xk = np.array([[0.                  , 0.               ],
                                    [0.                  , 0.               ],
                                    [0.                  , 0.               ],
                                    [1/(m+Xudot)         , 1/(m+Xudot)      ],
                                    [0.                  , 0.               ],
                                    [self.r/(Iz + Nrdot), -self.r/(Iz+Nrdot)]])
                
                grad_u = np.eye(2)

                w_tau = self.R @ tau
                u = np.linalg.pinv(self.B) @ w_tau

                gradxJ = 2 * (self.Q @ error)
                graduJ = 2 * (u)

                grad = delta_t * (grad_xk.transpose() @ gradxJ) + grad_u @ graduJ
                grad = grad.squeeze(-1)

                self.gradient_flag = grad
                """
                # Calculer le gradient du critère par rapport aux sorties du réseau
                grad = [
                    (-2/delta_t)*(alpha_x*alpha_x*(position[0]-self.target[0])*delta_t*self.robot.r*math.cos(position[2])
                    +alpha_y*alpha_y*(position[1]-self.target[1])*delta_t*self.robot.r*math.sin(position[2])
                    -alpha_teta*alpha_teta*(position[2]-self.target[2]-theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R)),

                    (-2/delta_t)*(alpha_x*alpha_x*(position[0]-self.target[0])*delta_t*self.robot.r*math.cos(position[2])
                    +alpha_y*alpha_y*(position[1]-self.target[1])*delta_t*self.robot.r*math.sin(position[2])
                    +alpha_teta*alpha_teta*(position[2]-self.target[2]-theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R))
                    ]
                """
                # Mettre à jour le moniteur avec le gradient actuel
                if self.monitor:
                    self.monitor.update(
                        position=position,
                        wheel_speeds=self.command,
                        gradient=grad,  # Utiliser le gradient qui vient d'être calculé
                        cost=crit_av
                    )
                
                # Stratégie d'apprentissage en fonction de l'évolution du critère
                if crit_ap <= crit_av:
                    # Effectuer une étape d'apprentissage
                    self.optimizer.zero_grad()
                    
                    # Convertir le gradient
                    grad_tensor = torch.tensor(grad, dtype=torch.float32)
                    
                    # Effectuer une étape d'apprentissage personnalisée
                    self.manual_backward(input_tensor, grad_tensor, 0.2, 0)
                else:
                    # Alternative si le critère ne s'améliore pas
                    # On peut soit ajouter du bruit soit quand même apprendre
                    self.optimizer.zero_grad()
                    
                    # Convertir le gradient
                    grad_tensor = torch.tensor(grad, dtype=torch.float32)
                    
                    # Effectuer une étape d'apprentissage personnalisée
                    self.manual_backward(input_tensor, grad_tensor, 0.2, 0)
        
        # Arrêter le robot à la fin de l'apprentissage
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
        outputs.backward(gradient=-grad_tensor)
        
        # Mise à jour des poids
        for param in self.network.parameters():
            if param.grad is not None:
                # Mise à jour manuelle avec le momentum si nécessaire
                param.data.add_(param.grad, alpha=-learning_rate)