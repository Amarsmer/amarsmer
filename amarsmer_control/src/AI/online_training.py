#!/usr/bin/env python3

import torch
import numpy as np
import sympy as sp
import custom_functions as cf


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def frameTransform(robot_coords, target_coords, frame='robot'):
    robot_coords = np.asarray(robot_coords).ravel()
    target_coords = np.asarray(target_coords).ravel()

    x_r, y_r, psi_r, _, _, _ = robot_coords
    x_t, y_t, psi_t, _, _, _ = target_coords

    cpsi_r, spsi_r = np.cos(psi_r), np.sin(psi_r)
    cpsi_t, spsi_t = np.cos(psi_t), np.sin(psi_t)

    if frame == 'robot':
        x = (x_t - x_r)*cpsi_r + (y_t - y_r)*spsi_r
        y = (y_t - y_r)*cpsi_r - (x_t - x_r)*spsi_r
        psi = psi_t - psi_r

    elif frame == 'target':
        x = (x_r - x_t)*cpsi_t + (y_r - y_t)*spsi_t
        y = (y_r - y_t)*cpsi_t - (x_r - x_t)*spsi_t
        psi = psi_r - psi_t

    return x, y, psi


class PyTorchOnlineTrainer:
    STATE_DIM = 7  # x, y, cos psi, sin psi, u, v, r

    def __init__(self, nn_model, in_learning_rate=5e-4, order=1,
                 in_Q=None, in_R=None, nb_thr=2, dt=0.05):

        self.network = nn_model
        self.nb_thr = nb_thr
        self.order = order
        self.dt = dt  # physical time step used for the gradient

        self.Q = np.eye(self.STATE_DIM) if in_Q is None else in_Q
        self.R = np.eye(nb_thr) if in_R is None else in_R

        self.training = True
        self.learning_rate = in_learning_rate
        self.input_coefficient = 40.0  # output scaling

        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.learning_rate)

        # Input history (for order > 1), filled on the first step
        self.input_list = [0.]*self.STATE_DIM*order
        self._history_empty = True

        # Outputs / monitoring
        self.u = np.zeros((nb_thr, 1))
        self.loss = None
        self.loss_display = np.zeros(2)
        self.gradient_display = np.zeros(nb_thr)
        self.error_display = None
        self.input_display = None
        self.state_display = None
        self.target_display = None
        self.target_frame = [0, 0, 0]
        self.trainer_set = False  # True once the first step has completed
        self.skew_display = None

        ## Modeling
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
        else:
            raise ValueError(f"Unsupported number of thrusters: {self.nb_thr}")

        mass, inertia, added_masses, viscous_drag, _ = cf.read_model()

        radius = 0.15
        hl = 0.3
        planar_added_mass = [added_masses[i] for i in [0, 1, 5]]
        planar_dampening = [viscous_drag[i] for i in [0, 1, 5]]

        self.compute_gradient, _, _, _ = cf.build_grad(B, mass, planar_added_mass, inertia[-1], planar_dampening,radius, hl, self.Q, self.R)

    def reframe(self, state_in, target_in):

        def theta_s(x, y):  # Angle skew, prevents the singularity in x=0
            if x == 0:
                x = 1e-10
            return np.tanh(20.*x**2)*np.arctan(1.5*y/x)

        self.state_display = state_in.copy()
        self.target_display = target_in.copy()

        s = state_in.ravel()

        dx_t, dy_t, dpsi_t = frameTransform(state_in, target_in, 'target')
        self.target_frame = [dx_t, dy_t, dpsi_t]

        state = np.array([dx_t,
                          dy_t,
                          np.cos(dpsi_t),
                          np.sin(dpsi_t),
                          s[3],
                          s[4],
                          s[5]]).reshape(-1, 1)

        angle_target = theta_s(dx_t, dy_t) if self.nb_thr == 2 else 0.

        return state, angle_target

    def compute_network_input(self, error):
        weight_matrix = np.diag([1/6, 1/6, 1, 1, 1, 1, 1/(2*np.pi)])
        return (weight_matrix @ error).ravel()

    def step(self, state, target):
        """One control + (optional) training step. Returns u as a 1D array."""
        state = np.asarray(state).reshape(-1, 1)
        target = np.asarray(target).reshape(-1, 1)

        # Error in the target frame
        state_cost, skew_angle = self.reframe(state, target)
        target_cost = np.zeros((self.STATE_DIM, 1))
        target_cost[2] = np.cos(skew_angle)
        target_cost[3] = np.sin(skew_angle)
        error = state_cost - target_cost

        self.skew_display = skew_angle.copy()

        network_input = self.compute_network_input(error)

        # Input history
        if self._history_empty:
            self.input_list = network_input.tolist()*self.order
            self._history_empty = False
        else:
            self.input_list = self.input_list[self.STATE_DIM:] + network_input.tolist()

        input_tensor = torch.tensor(self.input_list, dtype=torch.float32)

        # Forward pass
        if self.training:
            u_tensor = self.network(input_tensor)
        else:
            with torch.no_grad():
                u_tensor = self.network(input_tensor)

        # u_tensor = self.input_coefficient*u_tensor
        self.u = self.input_coefficient*u_tensor.detach().cpu().numpy().reshape(-1, 1)

        # Loss (monitoring, and automation in the node)
        crit_x = (error.T @ self.Q @ error).item()
        crit_u = (self.u.T @ self.R @ self.u).item()
        self.loss = crit_x + crit_u
        self.loss_display = np.array([crit_x, crit_u])

        grad = np.atleast_1d(self.compute_gradient(state_cost, target_cost, self.u, self.dt, 1/self.dt, 1/self.dt**2).squeeze())
        grad_tensor = torch.tensor(grad, dtype=torch.float32)
        self.gradient_display = grad.copy()

        # Training step
        if self.training:
            self.optimizer.zero_grad()
            u_tensor.backward(gradient=grad_tensor)
            self.optimizer.step()

        # Monitoring
        self.error_display = error.copy()
        self.input_display = network_input.copy()
        self.trainer_set = True

        return self.u.ravel()

    def reset(self, reset_optimizer=False):
        """Call after a discontinuity (e.g. robot teleported)."""
        self._history_empty = True
        if reset_optimizer:
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              lr=self.learning_rate)