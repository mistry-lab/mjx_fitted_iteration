import equinox
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from jax import config
from dataclasses import dataclass
from typing import Callable, Any
from functools import partial

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)


def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static)
        data_init: initial data state (pytree)
        qpos_init: initial positions (array)
        set_control_fn: fn(dx, u) -> dx to apply controls
        running_cost_fn: fn(dx, u) -> cost (float)
        terminal_cost_fn: fn(dx) -> cost (float)
        U: (N, nu) array of controls.

    Returns:
        states: (N, nq+nv) array of states
        total_cost: scalar total cost
    """
    def step_fn(dx, u):
        dx = set_control_fn(dx, u)
        dx = mjx.step(mx, dx)
        c = running_cost_fn(dx, u)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    dx_final, (states, costs) = jax.lax.scan(step_fn, dx0, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost


def make_loss(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn):
    """
    Create a loss function that only takes U as input.
    """
    def loss(U):
        _, total_cost = simulate_trajectory(
            mx, qpos_init,
            set_control_fn, running_cost_fn, terminal_cost_fn,
            U
        )
        return total_cost
    return loss


@dataclass
class TrajectoryOptimizer:
    """
    A flexible trajectory optimization framework.

    After initialization, you get `.loss` and `.grad_loss` as callables.
    """
    loss: Callable[[jnp.ndarray], float]
    grad_loss: Callable[[jnp.ndarray], jnp.ndarray]

    def gradient_descent(self, U0: jnp.ndarray, learning_rate=1e-2, tol=1e-6, max_iter=100):
        """
        Gradient descent on the control trajectory.

        U0: initial guess (N, nu)
        Returns: optimized U
        """
        U = U0
        for i in range(max_iter):
            g = self.grad_loss(U)
            U_new = U - learning_rate * g
            f_val = self.loss(U_new)
            print(f"Iteration {i}: cost={f_val}")
            if jnp.linalg.norm(U_new - U) < tol or jnp.isnan(g).any():
                return U_new
            U = U_new
        return U


if __name__ == "__main__":
    MODEL_XML = """
    <mujoco model="example">
        <option timestep="0.01" />
        <worldbody>
            <body name="object" pos="0 0 0">
                <joint type="free"/>
                <geom type="sphere" size="0.05"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    qpos_init = jnp.array([-0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    Nsteps, nu = 50, 1
    U0 = jnp.ones((Nsteps, nu)) * 2.0

    def set_control(dx, u):
        return dx.replace(qfrc_applied=jnp.zeros_like(dx.qfrc_applied).at[0].set(u[0]))

    def running_cost(dx, u):
        pos_error = dx.qpos[0] - 0.0
        return pos_error ** 2 + 1e-3 * jnp.sum(u ** 2)

    def terminal_cost(dx):
        return 10.0 * (dx.qpos[0] - 0.5) ** 2

    # Create the loss and grad_loss functions
    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    # Create the optimizer
    optimizer = TrajectoryOptimizer(loss=loss_fn, grad_loss=grad_loss_fn)

    # Run optimization
    optimal_U = optimizer.gradient_descent(U0, learning_rate=0.1, max_iter=100)
    print("Optimized trajectory:", optimal_U)
