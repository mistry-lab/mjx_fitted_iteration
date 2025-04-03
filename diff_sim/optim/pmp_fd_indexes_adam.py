import time
from dataclasses import dataclass
from typing import Callable, Any

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import mujoco.mjx as mjx 
 
# ------------------------------------------------------------------------------
# 1. An Equinox Module for storing B x T x nu controls and simulating them
# ------------------------------------------------------------------------------
class BatchTrajectory(eqx.Module):
    """
    Stores a batch of controls of shape (B, T, nu). 
    Handles simulating B parallel trajectories, each of length T, 
    and accumulating cost.
    """
    # Trainable parameter:
    controls: jnp.ndarray  # shape = (B, T, nu)
 
    # Static fields (not parameters). 
    # We store large objects or callables as eqx.static_field().
    mx: mjx.Model = eqx.static_field()
    qpos_init: jnp.ndarray
    set_ctrl_fn: Callable[[Any, jnp.ndarray], Any] = eqx.static_field()
    running_cost_fn: Callable[[Any], float] = eqx.static_field()
    terminal_cost_fn: Callable[[Any], float] = eqx.static_field()
    fd_cache: Any = eqx.static_field()  # If you need it
 
    def __call__(self) -> float:
        """
        Returns the scalar loss across the entire batch,
        e.g. the average cost of B parallel trajectories.
        """
        # We'll vmap over the batch dimension. 
        # For each b in [0..B-1], we do one trajectory simulation and 
        # compute a cost. Then we average or sum across the batch.
 
        B = self.controls.shape[0]
 
        def single_trajectory_loss(b_idx: int) -> float:
            """
            Simulate a single trajectory for the batch index b_idx.
            Return total cost (running + terminal).
            """
            # Create data, set initial position
            dx0 = mjx.make_data(self.mx)
            dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(self.qpos_init[b_idx]))
            dx0 = mjx.step(self.mx, dx0)  # initial sync
 
            # We store states and costs across T steps
            # But if you only need total cost, we can accumulate directly.
 
            def scan_body(dx, t):
                # pick control from self.controls
                u = self.controls[b_idx, t]  # shape (nu,)
                dx_with_ctrl = self.set_ctrl_fn(dx, u)
                dx_next = mjx.step(self.mx, dx_with_ctrl)
 
                cost_t = self.running_cost_fn(dx_next)
                return dx_next, cost_t
 
            T = self.controls.shape[1]
            dx_final, costs = jax.lax.scan(scan_body, dx0, jnp.arange(T))
            total_cost = jnp.sum(costs) + self.terminal_cost_fn(dx_final)
            return total_cost
 
        # Vectorize over b_idx in [0..B)
        costs_b = jax.vmap(single_trajectory_loss)(jnp.arange(B))
 
        # Return average or sum across batch. Adjust to your preference:
        return jnp.mean(costs_b)
 
 
# ------------------------------------------------------------------------------
# 2. Building a "make_batch_loss_module" style constructor (optional)
# ------------------------------------------------------------------------------
def make_batch_loss_module(
    mx,
    qpos_init: jnp.ndarray,
    set_ctrl_fn: Callable[[Any, jnp.ndarray], Any],
    running_cost_fn: Callable[[Any], float],
    terminal_cost_fn: Callable[[Any], float],
    fd_cache: Any,
    B: int,
    T: int,
    nu: int,
    key: jax.random.PRNGKey
) -> BatchTrajectory:
    """
    Utility to create a BatchTrajectory module with random initialization 
    for the controls of shape (B, T, nu).
    """
    # Example init for controls
    init_controls = 0.1 * jax.random.normal(key, (B, T, nu))
    return BatchTrajectory(
        controls=init_controls,
        mx=mx,
        qpos_init=qpos_init,
        set_ctrl_fn=set_ctrl_fn,
        running_cost_fn=running_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        fd_cache=fd_cache,
    )
 

# 4.5: Evaluate or visualize one trajectory (say b=0)
# or pick random b, etc.
# We'll define a "simulate_trajectory" style function manually:
@jax.jit
def simulate_trajectory_for_b(m: BatchTrajectory, b_idx: int):
    dx0 = mjx.make_data(m.mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(m.qpos_init[b_idx]))
    dx0 = mjx.step(m.mx, dx0)

    def scan_body(dx, t):
        u = m.controls[b_idx, t]
        dx_next = mjx.step(m.mx, m.set_ctrl_fn(dx, u))
        state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return dx_next, state

    T_ = m.controls.shape[1]
    _, states = jax.lax.scan(scan_body, dx0, jnp.arange(T_))
    return states  # shape = (T_, qpos_dim+qvel_dim)

 
# ------------------------------------------------------------------------------
# 3. Example training loop using Optax + Adam
# ------------------------------------------------------------------------------
def train_batch_trajectories(
    model_module: BatchTrajectory,
    num_steps: int = 1000,
    lr: float = 1e-3
) -> BatchTrajectory:
    """
    Example training loop that runs an Adam optimizer on the 
    batch trajectory problem.
    """
    # 1) Create the optimizer and its state
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model_module, eqx.is_array))
 
    # 2) Define a loss-and-grad function
    # We will jit+grad the model's __call__:
    @jax.jit
    def loss_and_grad(model: BatchTrajectory):
        loss_val = model()  # calls the module => returns the cost
        grads = jax.grad(lambda m: m())(model)
        return loss_val, grads
 
    # 3) Training loop
    m = model_module
    for step_idx in range(num_steps):
        now = time.time()
        loss_val, grads = loss_and_grad(m)
        updates, opt_state_ = optimizer.update(
            grads, opt_state, params=m
        )
        # Apply the updates
        m = eqx.apply_updates(m, updates)
        opt_state = opt_state_
        print(f"Time: {time.time() - now}")
        print(f"Loss={loss_val:0.6f}")
        print(f"\n--- Iteration {step_idx} ---")
 
        if step_idx % 50 == 0:
            print(f"Step={step_idx}, Loss={loss_val:0.6f}")
 
    return m
 