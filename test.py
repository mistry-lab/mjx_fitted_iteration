import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
# Initialize devices and model
devices = jax.devices()
is_gpu = next((d for d in devices if 'gpu' in d.device_kind.lower()), None)

model = mujoco.MjModel.from_xml_path('/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml')
mx = mjx.put_model(model, device=is_gpu)


def init_state(xs, mx, dx):
    qp = xs[..., :mx.nq]
    qv = xs[..., mx.nq:]
    qpos = dx.qpos.at[:].set(qp)
    qvel = dx.qvel.at[:].set(qv)
    dx_n = dx.replace(
        qpos=qpos, qvel=qvel
    )
    return dx_n


def make_data(m):
    return mjx.make_data(m)


def sim_traj(x_init, m, nsteps):
    dx = make_data(m)
    qp = x_init[:m.nq]
    qv = x_init[m.nq:]
    qpos = dx.qpos.at[:].set(qp)
    qvel = dx.qvel.at[:].set(qv)
    dx_n = dx.replace(
        qpos=qpos, qvel=qvel
    )

    def sim_step(dx, _):
        mjx_data = mjx.step(m, dx)
        return mjx_data, jnp.stack([mjx_data.qpos, mjx_data.qvel])

    _, traj = jax.lax.scan(
        sim_step, dx_n, None, length=nsteps
    )
    return traj


n_timesteps = 100
n_simulations = 1000
key = jax.random.PRNGKey(0)
pole = jax.random.normal(key, shape=(n_simulations,)) * 0.1
cart = jnp.zeros(n_simulations)

# the position
qpos = jnp.stack([cart, pole], axis=-1)
qvel = jnp.zeros((n_simulations, 2))
x_init = jnp.concatenate([qpos, qvel], axis=-1)

# JIT compilation of the vectorized simulation function
vmap_func = jax.vmap(sim_traj, in_axes=(0, None, None))
jit_vectorized_simulate = jax.jit(vmap_func, static_argnums=(2,))

# Execute the simulation
pole_positions = jit_vectorized_simulate(x_init, mx, n_timesteps)
print(pole_positions.shape)

