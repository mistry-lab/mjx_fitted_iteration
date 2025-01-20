import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import equinox
import optax
from diff_sim.traj_opt.policy import (
    simulate_trajectories, make_loss_multi_init, make_step_fn, build_fd_cache
)
from diff_sim.nn.base_nn import Network

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path("../../xmls/finger_mjx.xml")
    mx = mjx.put_model(model)
    dx_template = mjx.make_data(mx)
    dx_template = jax.tree.map(upscale, dx_template)

    # Suppose we define a batch of initial conditions
    # e.g. 5 different qpos, qvel
    qpos_inits = jnp.array([
        [-0.8,  0.0, -0.8]
    ])
    # repeat qpos_inits to match to size 164
    qpos_inits = jnp.repeat(qpos_inits, 64, axis=0)
    qpos_inits += 0.01 * jax.random.normal(jax.random.PRNGKey(0), qpos_inits.shape)
    qvel_inits = jnp.zeros_like(qpos_inits)  # or any distribution you like

    Nsteps, nu = 300, 2

    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.002 * jnp.sum(u ** 2) + 0.001 * pos_finger ** 2

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 4 * pos_finger ** 2

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    class PolicyNet(Network):
        layers: list
        act: callable

        def __init__(self, dims: list, key):
            keys = jax.random.split(key, len(dims))
            self.layers = [equinox.nn.Linear(
                dims[i], dims[i + 1], key=keys[i], use_bias=True
            ) for i in range(len(dims) - 1)]
            self.act = jax.nn.relu

        def __call__(self, x, t):
            for layer in self.layers[:-1]:
                x = self.act(layer(x))
            x = self.layers[-1](x).squeeze()
            x = jnp.tanh(x) * 1
            return x

    # Create the new multi-initial-condition loss function
    loss_fn = make_loss_multi_init(
        mx,
        qpos_inits,     # shape (B, n_qpos)
        qvel_inits,     # shape (B, n_qpos) or (B, n_qvel)
        set_control_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        length=Nsteps,
        reduce_cost="mean",
    )

    # JIT the gradient

    # Create your FD cache (if you want it explicitly) or rely on the above
    # fd_cache = build_fd_cache(dx_template, jnp.zeros((mx.nu,)), ...)

    # Create your policy net, optimizer, and do gradient descent
    nn = PolicyNet([6, 64, 64, 2], key=jax.random.PRNGKey(0))
    adam = optax.adamw(8e-4)
    opt_state = adam.init(equinox.filter(nn, equinox.is_array))

    # Same "Policy" class as before
    from diff_sim.traj_opt.policy import Policy
    optimizer = Policy(loss=loss_fn)
    optimal_nn = optimizer.solve(nn, adam, opt_state, max_iter=75)

    fd_cache = build_fd_cache(dx_template)
    step_fn = make_step_fn(mx, set_control, fd_cache)
    # Evaluate final performance *on the entire batch*
    params, static = equinox.partition(optimal_nn, equinox.is_array)
    states_batched, cost_batched = simulate_trajectories(
        mx, qpos_inits, qvel_inits,
        running_cost, terminal_cost,
        step_fn=step_fn,
        params=params,
        static=static,
        length=Nsteps,
        reduce_cost="mean",
    )

    # visualize the trajectories
    from diff_sim.utils.mj import visualise_traj_generic
    data = mujoco.MjData(model)
    visualise_traj_generic(jnp.array(states_batched), data, model)