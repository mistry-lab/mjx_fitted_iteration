import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import equinox
import optax

# Assume these imports point to the optimized versions above
from diff_sim.traj_opt.policy import Policy, make_loss, make_step_fn, build_fd_cache
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
    model = mujoco.MjModel.from_xml_path("../../xmls/finger_mjx.xml")
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree_map(upscale, dx)
    qpos_init = jnp.array([-0.8, 0, -.8])
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

    # Make the loss and gradient
    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost, length=Nsteps)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    # Build the FD cache (usually not needed if you let make_loss handle it, but shown for clarity)
    fd_cache = build_fd_cache(
        dx, jnp.zeros((mx.nu,)),
        target_fields={'qpos', 'qvel', 'ctrl'},
        eps=1e-6
    )
    step_fn = make_step_fn(mx, set_control, fd_cache)

    nn = PolicyNet([6, 64, 64, 2], key=jax.random.PRNGKey(0))

    adam = optax.adamw(1e-3)
    opt_state = adam.init(equinox.filter(nn, equinox.is_array))

    optimizer = Policy(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_nn = optimizer.solve(nn, adam, opt_state, max_iter=13)

    # Then you can visualize or evaluate the final trajectory:
    from diff_sim.utils.mj import visualise_traj_generic
    from diff_sim.traj_opt.policy import simulate_trajectory

    params, static = equinox.partition(optimal_nn, equinox.is_array)
    d = mujoco.MjData(model)
    x, cost = simulate_trajectory(mx, qpos_init, running_cost, terminal_cost, step_fn, params, static, Nsteps)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
