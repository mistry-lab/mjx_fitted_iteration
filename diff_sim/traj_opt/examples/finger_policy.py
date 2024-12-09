import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import equinox
from diff_sim.traj_opt.policy import Policy, make_loss
from diff_sim.nn.base_nn import Network
import optax

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":
    path = "../../xmls/finger_mjx.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    qpos_init = jnp.array([1, 0, -.8])
    Nsteps, nu = 600, 1
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 0

    def set_control(dx, nn):
        x = jnp.concatenate([dx.qpos, dx.qvel])
        u = nn(x, dx.time)
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.01 * pos_finger ** 2 + jnp.sum(u ** 2)

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 1 * pos_finger ** 2


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
            return x


    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost, length=600)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))
    nn = PolicyNet([6, 32, 32, 1], key=jax.random.PRNGKey(0))
    adam = optax.adamw(4e-3)
    opt_state = adam.init(equinox.filter(nn, equinox.is_array))

    optimizer = Policy(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_nn = optimizer.solve(nn, adam, opt_state, max_iter=35)

    from diff_sim.utils.mj import visualise_traj_generic
    from diff_sim.traj_opt.policy import simulate_trajectory
    import mujoco

    params, static = equinox.partition(optimal_nn, equinox.is_array)
    d = mujoco.MjData(model)
    x, cost = simulate_trajectory(mx, qpos_init, set_control, running_cost, terminal_cost, params, static, 600)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
