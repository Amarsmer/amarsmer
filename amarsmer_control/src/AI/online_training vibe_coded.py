#!/usr/bin/env python3
import time
import math
import torch
import numpy as np
from typing import Optional, Iterable

def theta_s(x, y):
    return math.tanh(10.0 * x) * math.atan(1.0 * y)

class PyTorchOnlineTrainer:
    """
    Physics-informed online trainer for a Pioneer-like differential robot.
    - robot: object with attributes/methods:
        .current_pose (indexable, contains [x, y, theta])
        .current_twist (indexable, contains [vx, vy, vtheta])
        .mass, .added_masses (indexable), .inertia (indexable)
        .move(cmd_list, rest_list)
    - nn_model: PyTorch module mapping 6-dim normalized error -> 2 wheel commands (u_left, u_right)
    - monitor: optional object with set_target(target) and update(...) for logging/visualization
    """
    def __init__(
        self,
        robot,
        nn_model: torch.nn.Module,
        monitor=None,
        Q: Optional[np.ndarray]=None,
        R: Optional[np.ndarray]=None,
        lr: float = 1e-4,
        grad_clip: float = 1.0,
        input_scales: Optional[Iterable[float]] = None,
        wheel_command_scale: float = 10.0,
        update_on_improve: bool = False,
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
                           [self.r, -self.r]], dtype=np.float32)   # 3x2
        self.B_torch = torch.tensor(self.B, dtype=torch.float32)

        # optimizer & hyperparams
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, momentum=0.0)
        self.grad_clip = grad_clip
        self.wheel_command_scale = wheel_command_scale
        self.update_on_improve = bool(update_on_improve)

        # input scaling (normalize the 6 inputs)
        if input_scales is None:
            # good starting defaults: positions ~ meters -> divide by 10, angle by pi, velocities raw
            self.input_scales = np.array([10.0, 10.0, math.pi, 1.0, 1.0, 1.0], dtype=np.float32)
        else:
            self.input_scales = np.array(input_scales, dtype=np.float32)

        # runtime flags/state
        self.running = False      # main loop runs while True
        self.training = False     # whether parameter updates are done
        self.command_set = False
        self.target = None
        self.last_output_grad = None
        self.last_cost = None


    def _read_state(self):
        """Return state (6x1 numpy): [x,y,theta, vx,vy,vtheta] as column vector."""
        position = np.array([self.robot.current_pose[i] for i in [0, 1, 5]], dtype=np.float32).reshape(-1, 1)
        speed = np.array([self.robot.current_twist[i] for i in [0, 1, 5]], dtype=np.float32).reshape(-1, 1)
        state = np.vstack([position, speed])
        return state

    def _make_network_input(self, error: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Build 6-dim input: [ex, ey, e_theta - theta_s(x,y), evx, evy, evtheta]
        error and state are 6x1 numpy arrays.
        Returns normalized 1D numpy of length 6.
        """
        x_err = float(error[0])
        y_err = float(error[1])
        theta_err = float(error[2] - theta_s(float(state[0]), float(state[1])))
        vx_err = float(error[3])
        vy_err = float(error[4])
        vtheta_err = float(error[5])
        raw = np.array([x_err, y_err, theta_err, vx_err, vy_err, vtheta_err], dtype=np.float32)
        return (raw / self.input_scales).astype(np.float32)

    def stop(self):
        """Stop the trainer loop (safe to call from another thread)."""
        self.running = False

    def train(self, target: Optional[Iterable[float]] = None, training: bool = True):
        """
        Main loop of the trainer. Signature kept so you can do:
            threading.Thread(target=trainer.train, args=(target, True)).start()
        - target: 6-element iterable (pose [x,y,theta], twist [vx,vy,vtheta]) or None if already set
        - training: whether to actually update network parameters
        """
        if target is not None:
            self.target = np.array(target, dtype=np.float32).reshape(-1, 1)
        if self.target is None:
            raise ValueError("Target must be provided to train(target, ...)")

        if self.monitor:
            try:
                self.monitor.set_target(self.target)
            except Exception:
                pass

        self.training = bool(training)
        self.running = True

        # prepare torch versions of Q and R
        Q_torch = torch.tensor(self.Q, dtype=torch.float32)
        R_torch = torch.tensor(self.R, dtype=torch.float32)

        try:
            while self.running:
                loop_start = time.time()

                # read state & error
                state = self._read_state()                       # 6x1 numpy
                error = state - np.array(self.target).reshape(-1, 1)

                # build network input
                net_in = self._make_network_input(error, state)  # length-6 numpy
                input_tensor = torch.tensor(net_in, dtype=torch.float32)  # shape (6,)

                # FORWARD: get network outputs (unscaled)
                # ensure forward is done with grad enabled only when needed
                if self.training:
                    outputs = self.network(input_tensor)   # e.g. shape (2,) or (2,1)
                else:
                    with torch.no_grad():
                        outputs = self.network(input_tensor)

                # normalize outputs shape to 1D tensor of length 2
                if outputs.ndim > 1:
                    outputs = outputs.view(-1)
                u_tensor = outputs.view(-1)   # shape (2,)

                # compute tau = B @ u (3x2 @ 2x1 => 3x1)
                tau = self.B_torch @ u_tensor.view(-1, 1)   # (3,1)

                # compute current cost (scalar): error^T Q error + tau^T R tau
                error_torch = torch.tensor(error.reshape(-1, 1), dtype=torch.float32)   # (6,1)
                term_e = (error_torch.T @ Q_torch @ error_torch).squeeze()
                term_tau = (tau.T @ R_torch @ tau).squeeze()
                cost = (term_e + term_tau).detach().cpu().item()   # float for logging

                # send command to robot (scale outputs to wheel speeds)
                u_np = u_tensor.detach().cpu().numpy().reshape(-1, 1)   # (2,1)
                # mapping: assume network outputs [left, right] (if reversed, swap)
                left_speed = float(u_np[0] * self.wheel_command_scale)
                right_speed = float(u_np[1] * self.wheel_command_scale)
                try:
                    self.robot.move([left_speed, right_speed, 0, 0], [0 for _ in range(1, 5)])
                    self.command = u_np
                    self.command_set = True
                except Exception:
                    # robot move failed; continue loop but don't crash
                    pass

                # small sleep to maintain loop rate (original 50 ms)
                time.sleep(0.05)

                # read new state/error after motion (to compute physics-informed gradient)
                new_state = self._read_state()
                new_error = new_state - np.array(self.target).reshape(-1, 1)

                # recompute network input after motion (we'll use the same network to compute output for gradient)
                net_in_after = self._make_network_input(new_error, new_state)
                input_tensor_after = torch.tensor(net_in_after, dtype=torch.float32)

                # compute tau (using network output after motion)
                # For gradient computation we need numeric tau and error (numpy) plus physics params
                with torch.no_grad():
                    outputs_after = self.network(input_tensor_after)
                    if outputs_after.ndim > 1:
                        outputs_after = outputs_after.view(-1)
                    u_after = outputs_after.detach().cpu().numpy().reshape(-1, 1)   # (2,1)
                    tau_after = (self.B @ u_after).astype(np.float32)             # (3,1) numpy

                # compute cost after motion (for improvement check & monitor)
                error_np = new_error  # (6,1) numpy
                crit_after = float((error_np.T @ self.Q @ error_np + tau_after.T @ self.R @ tau_after).squeeze())

                # Physics-informed gradient calculation (numpy)
                delta_t = max(1e-6, time.time() - loop_start)  # avoid zero
                grad = self._compute_physics_gradient_np(error_np, tau_after, delta_t)

                # Convert grad from numpy (2,) to torch tensor shaped like network output
                grad_tensor = torch.tensor(grad, dtype=torch.float32)
                if grad_tensor.ndim == 0:
                    grad_tensor = grad_tensor.view(1)
                # ensure grad has same shape as outputs_after
                # outputs_after is a torch tensor; we want gradient vector of same shape (1D)
                if outputs_after.ndim > 1:
                    grad_tensor = grad_tensor.view_as(outputs_after)
                else:
                    grad_tensor = grad_tensor.view(-1)

                # Optionally check improvement
                do_update = True
                if self.update_on_improve:
                    if self.last_cost is None or crit_after <= self.last_cost:
                        do_update = True
                    else:
                        do_update = False

                # Training update: inject gradient wrt network outputs (physics-based)
                if self.training and do_update:
                    # Zero grads
                    self.optimizer.zero_grad()

                    # Forward on the input_tensor_after to get outputs as part of autograd graph
                    outputs_for_back = self.network(input_tensor_after)
                    if outputs_for_back.ndim > 1:
                        outputs_for_back = outputs_for_back.view(-1)

                    # Important: outputs_for_back must require grad (it will if network parameters require_grad True)
                    # Backprop injecting -grad (we use negative because we want to descend the cost)
                    # Make sure shapes align: grad_tensor should be same shape as outputs_for_back
                    # Use .detach() to ensure grad_tensor is not part of graph
                    try:
                        outputs_for_back.backward(gradient=-grad_tensor.detach())
                    except RuntimeError as e:
                        # fallback: try reshape adjustments
                        try:
                            outputs_for_back.backward(gradient=-grad_tensor.detach().view_as(outputs_for_back))
                        except Exception:
                            # if backward still fails, skip update this step
                            do_update = False

                    if do_update:
                        # Clip and step
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                        # Some networks may have None grads for some params; optimizer.step() handles that
                        self.optimizer.step()

                    # store last_output_grad for monitor (numpy column)
                    try:
                        self.last_output_grad = grad_tensor.detach().cpu().numpy().reshape(-1, 1)
                    except Exception:
                        self.last_output_grad = None

                # monitor update
                if self.monitor:
                    try:
                        self.monitor.update(
                            position=new_state,
                            wheel_speeds=u_np,
                            gradient=self.last_output_grad,
                            cost=crit_after
                        )
                    except Exception:
                        pass

                # update last cost
                self.last_cost = crit_after

            # end while running
        finally:
            # ensure robot stops and running is false
            try:
                self.robot.move([0, 0, 0, 0], [0 for _ in range(1, 5)])
            except Exception:
                pass
            self.running = False

    def _compute_physics_gradient_np(self, error_np: np.ndarray, tau_np: np.ndarray, delta_t: float) -> np.ndarray:
        """
        Recreate the physics-based gradient computation you had in the original code,
        returning a (2,) numpy vector representing dJ/du (matching network outputs order).
        - error_np: (6,1) numpy
        - tau_np: (3,1) numpy
        """
        # read robot inertial properties
        m = float(getattr(self.robot, "mass", 1.0))
        added = getattr(self.robot, "added_masses", None)
        if added is None:
            Xudot = 0.0
            Nrdot = 0.0
        else:
            Xudot = float(added[0])
            Nrdot = float(added[5])
        inertia = getattr(self.robot, "inertia", None)
        Iz = float(inertia[-1]) if inertia is not None else 1.0

        # grad_xk: effect of wheel inputs on state derivative (6x2)
        grad_xk = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0 / (m + Xudot), 1.0 / (m + Xudot)],
            [0.0, 0.0],
            [self.r / (Iz + Nrdot), -self.r / (Iz + Nrdot)]
        ], dtype=np.float32)   # shape (6,2)

        grad_u = np.eye(2, dtype=np.float32)   # shape (2,2)

        # w_tau = R @ tau  (R is 3x3, tau is 3x1)
        w_tau = (self.R @ tau_np).astype(np.float32)   # (3,1)

        # pseudo u = pinv(B) @ w_tau  (2x3 @ 3x1 -> 2x1)
        # Using pseudo-inverse because original code did
        try:
            u_pinv = np.linalg.pinv(self.B).astype(np.float32)  # (2,3)
            u_like = (u_pinv @ w_tau).astype(np.float32)        # (2,1)
        except Exception:
            u_like = np.zeros((2,1), dtype=np.float32)

        # gradxJ = 2 * Q @ error  -> shape (6,1)
        gradxJ = 2.0 * (self.Q @ error_np).astype(np.float32)  # (6,1)

        # graduJ = 2 * u_like  -> (2,1)
        graduJ = 2.0 * u_like   # (2,1)

        # final gradient (2,1) => delta_t * (grad_xk^T @ gradxJ) + grad_u @ graduJ
        part1 = (grad_xk.T @ gradxJ).astype(np.float32)   # (2,1)
        part2 = (grad_u @ graduJ).astype(np.float32)      # (2,1)
        grad = (delta_t * part1 + part2).astype(np.float32)  # (2,1)

        return grad.squeeze(-1)   # shape (2,)