import jax
import jax.numpy as jnp
import optax
import equinox
import time
from pydantic.dataclasses import dataclass
from typing import Callable
from diff_sim.optim.meta_context import Context
from mujoco import mjx
from diff_sim.optim.ilqr import simulate_trajectory_ilqr

def make_pmp_step(
    step_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
    ctx: Context
):
    """
    Builds a single-step update function for the PMP solver using Adam.
    Returns a function pmp_step(x0, U, xdes, opt_state, optimizer) -> (U_new, cost_new, opt_state_new).
    """
    @equinox.filter_jit
    def total_cost(U, x0, xdes):
        """
        Forward-simulate for the entire horizon and sum the running + terminal cost.
        """
        X, _, C = simulate_trajectory_ilqr(x0, U, xdes, step_fn, ctx)
        return jnp.sum(C)

    @equinox.filter_jit
    def pmp_step(x0, U, xdes, opt_state, optimizer):
        """
        Single iteration of the PMP solver:
          1) Compute gradient of total cost wrt U.
          2) Update U with Adam.
          3) Return new U, new cost, and updated opt_state.
        """
        def cost_fn(U_):
            return total_cost(U_, x0, xdes)

        grads = jax.grad(cost_fn)(U)
        updates, new_opt_state = optimizer.update(grads, opt_state, U)
        U_new = optax.apply_updates(U, updates)

        cost_new = cost_fn(U_new)
        return U_new, cost_new, new_opt_state

    return pmp_step


@dataclass
class PMP:
    """
    A class-based PMP solver, mirroring the iLQR class structure.
    """
    pmp_step: Callable  # (x0, U, xdes, opt_state, optimizer) -> (U_new, cost_new, opt_state_new)

    def solve(
        self,
        X0: jnp.ndarray,       # shape (batch, nq) or similar
        U0: jnp.ndarray,       # shape (batch, nsteps, nu)
        xdes: jnp.ndarray,     # shape (batch, ?)
        tol: float = 1e-5,
        max_iter: int = 50,
        lr: float = 1e-2
    ):
        """
        Solves for the optimal control sequence using Adam-based PMP.
        - X0: initial states (batch x ...)
        - U0: initial guess for controls (batch x nsteps x nu)
        - xdes: desired target states (batch x ...)
        - tol: convergence tolerance
        - max_iter: max iterations
        - lr: learning rate for Adam
        Returns: (U_final, cost_final_mean)
        """
        batch_size = X0.shape[0]

        # 1) Initialize the Adam optimizer
        optimizer = optax.adam(learning_rate=lr)

        # 2) Build an optimizer state for each element of the batch
        def init_opt_state_fn(U_i):
            return optimizer.init(U_i)

        opt_state = jax.vmap(init_opt_state_fn)(U0)

        # 3) Initialize iteration variables
        U = U0
        prev_cost_mean = jnp.inf
        prev_cost = jnp.inf * jnp.ones(X0.shape[0])
        total_cost_mean = jnp.inf

        for i in range(max_iter):
            now = time.time()
            U_new, C, opt_state_new = jax.vmap(self.pmp_step, in_axes=(0,0,0,0,None))(
                X0, U, xdes, opt_state, optimizer
            )
            total_cost_mean = jnp.mean(C)
            print(f"\nIteration {i} :")
            print(f"Computing time: {time.time() - now}")
            print(f"Iteration {i}: Average cost={total_cost_mean}")

            improvement_mean = prev_cost_mean - total_cost_mean
            improvement = prev_cost - C

            if improvement_mean < 0:
                pass

            # Convergence check by improvement
            if jnp.abs(improvement_mean) < tol:
                print(f"Converged at iteration {i} with cost={float(total_cost_mean):0.6f}")
                U = U_new
                break

            print(f"Trajectory optimized: {jnp.sum(jnp.abs(improvement) < tol)}/ {len(improvement)}")

            # Check the norm of the control update
            ctrl_diff_norm = jnp.linalg.norm(U_new - U)
            if ctrl_diff_norm < tol:
                print(f"Control update norm below tolerance at iteration {i}")
                U = U_new
                break

            # Update for next iteration
            U = U_new
            opt_state = opt_state_new
            prev_cost_mean = total_cost_mean

        return U, total_cost_mean
