#!/usr/bin/env python3
import time
import math
import torch
import numpy as np
from typing import Optional, Iterable, Deque
from collections import deque

def theta_s(x, y):
    return math.tanh(10.0 * x) * math.atan(1.0 * y)

class PyTorchOnlineTrainer:
    """
    Physics-informed online trainer for a Pioneer-like differential robot.
    Uses hand-computed physics gradient (dJ/du) per step.
    """
    def __init__(
        self,
        robot,
        nn_model: torch.nn.Module,
        monitor=None,
        Q: Optional[np.ndarray]=None,
        R: Optional[np.ndarray]=None,
        lr: float = 1e-4,
        grad_clip: float = 0.5,
        input_scales: Optional[Iterable[float]] = None,
        wheel_command_scale: float = 10.0,
        output_limit: float = 1.0,
        replay_size: int = 32,
    ):
        self.robot = robot
        self.network = nn_model
        self.monitor = monitor

        # cost weights
        self.Q = Q if Q is not None else np.eye(6, dtype=np.float32)
        self.R = R if R is not None else np.eye(3, dtype=np.float32)

        # vehicle geometry
        self.r = 0.15
        self.B = np.array([[1.0, 1.0],
                           [0.0, 0.0],
                           [self.r, -self.r]], dtype=np.float32)
        self.B_torch = torch.tensor(self.B, dtype=torch.float32)

        # optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-6)
        self.grad_clip = grad_clip
        self.wheel_command_scale = wheel_command_scale
        self.output_limit = output_limit

        # input scaling
        if input_scales is None:
            self.input_scales = np.array([10.0, 10.0, math.pi, 1.0, 1.0, 1.0], dtype=np.float32)
        else:
            self.input_scales = np.array(input_scales, dtype=np.float32)

        # replay buffer
        self.replay_buffer: Deque[tuple] = deque(maxlen=replay_size)

        # runtime state
        self.running = False
        self.training = False
        self.target = None
        self.last_output_grad = None
        self.last_cost = None
        self.command_set = False

    def _read_state(self) -> np.ndarray:
        pos = np.array([self.robot.current_pose[i] for i in [0,1,5]], dtype=np.float32).reshape(-1,1)
        vel = np.array([self.robot.current_twist[i] for i in [0,1,5]], dtype=np.float32).reshape(-1,1)
        return np.vstack([pos, vel])

    def _make_network_input(self, error: np.ndarray, state: np.ndarray) -> np.ndarray:
        theta_err = float(error[2] - theta_s(float(state[0]), float(state[1])))
        raw = np.array([
            float(error[0]), float(error[1]), theta_err,
            float(error[3]), float(error[4]), float(error[5])
        ], dtype=np.float32)
        return raw / self.input_scales

    def stop(self):
        self.running = False

    def _compute_hand_gradient(self, error: np.ndarray, tau: np.ndarray, delta_t: float) -> np.ndarray:
        """Compute hand physics gradient exactly as specified."""
        m = self.robot.mass
        Xudot = self.robot.added_masses[0]
        Nrdot = self.robot.added_masses[5]
        Iz = self.robot.inertia[-1]

        grad_xk = np.array([[0., 0.],
                            [0., 0.],
                            [0., 0.],
                            [1./(m + Xudot), 1./(m + Xudot)],
                            [0., 0.],
                            [self.r/(Iz + Nrdot), -self.r/(Iz + Nrdot)]], dtype=np.float32)

        grad_u = np.eye(2, dtype=np.float32)

        w_tau = self.R @ tau
        u_like = np.linalg.pinv(self.B) @ w_tau

        gradxJ = 2 * (self.Q @ error)
        graduJ = 2 * u_like

        grad = delta_t * (grad_xk.T @ gradxJ) + grad_u @ graduJ
        return grad.squeeze(-1)  # shape (2,)

    def train(self, target: Optional[Iterable[float]] = None, training: bool = True):
        if target is not None:
            self.target = np.array(target, dtype=np.float32).reshape(-1,1)
        if self.target is None:
            raise ValueError("Target must be provided.")

        self.training = training
        self.running = True
        device = next(self.network.parameters()).device

        if self.monitor:
            try:
                self.monitor.set_target(self.target)
            except Exception:
                pass

        try:
            debut = time.time()
            while self.running:
                loop_start = time.time()
                delta_t = max(1e-6, loop_start - debut)
                debut = loop_start

                # --- read state ---
                state = self._read_state()
                error = state - self.target

                # --- forward ---
                net_in = self._make_network_input(error, state)
                input_tensor = torch.tensor(net_in, dtype=torch.float32, device=device).unsqueeze(0)
                self.network.eval()
                with torch.no_grad():
                    outputs = self.network(input_tensor).view(-1)
                outputs_clipped = torch.clamp(outputs, -self.output_limit, self.output_limit)
                u_np = (outputs_clipped.detach().cpu().numpy().reshape(-1,1) * self.wheel_command_scale)

                # --- send command ---
                try:
                    left_speed, right_speed = float(u_np[0]), float(u_np[1])
                    self.robot.move([left_speed, right_speed, 0,0], [0]*4)
                    self.command_set = True
                except Exception:
                    pass

                # --- read new state ---
                time.sleep(0.05)
                new_state = self._read_state()
                new_error = new_state - self.target

                # compute tau after motion
                tau = self.B @ u_np.astype(np.float32)

                # --- hand-computed gradient ---
                grad_np = self._compute_hand_gradient(new_error, tau, delta_t)
                grad_tensor = torch.tensor(grad_np, dtype=torch.float32, device=device).view(-1)
                grad_tensor = torch.clamp(grad_tensor, -5.0, 5.0)
                self.last_output_grad = grad_tensor.detach().cpu().reshape(-1,1)

                # --- optimizer step ---
                if self.training:
                    self.optimizer.zero_grad()
                    input_after_t = torch.tensor(self._make_network_input(new_error, new_state),
                                                 dtype=torch.float32, device=device).unsqueeze(0)
                    outputs_for_back = self.network(input_after_t).view(-1)
                    outputs_for_back.backward(gradient=-grad_tensor)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                    self.optimizer.step()

                # --- monitor ---
                if self.monitor:
                    try:
                        crit_after = float((new_error.T @ self.Q @ new_error + tau.T @ self.R @ tau).squeeze())
                        self.monitor.update(
                            position=new_state,
                            wheel_speeds=u_np,
                            gradient=self.last_output_grad,
                            cost=crit_after
                        )
                    except Exception:
                        pass

        finally:
            try:
                self.robot.move([0,0,0,0], [0]*4)
            except Exception:
                pass
            self.running = False
