
import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network

model_path = os.path.join(os.path.dirname(__file__), '../xmls/point_mass_tendon.xml')
def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

_cfg = Config(
    lr=4e-3,
    seed=4,
    batch=240,
    samples=1,
    epochs=1000,
    eval=50,
    num_gpu=1,
    dt=0.01,
    ntotal=512,
    nsteps=16,
    mx=mjx.put_model(gen_model()),
    gen_model=gen_model
)

class Policy(Network):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(
            dims[i], dims[i + 1], key=keys[i], use_bias=True
        ) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

    def __call__(self, x, t):
        # t = t if t.ndim == 1 else t.reshape(1)
        # x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x).squeeze()
        return x

def policy(net: Network, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = state_encoder(mx, dx)
    t = jnp.expand_dims(dx.time, axis=0)
    u = net(x, t)
    dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
    return dx, u

def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    ball_pos = dx.qpos[3:5]
    # mocap pos for some reason has one leading dimension (inherent to mujoco)
    pos_diff = ball_pos - dx.mocap_pos[0, :2]
    joint_pos = dx.qpos[:3]
    joint_vel = dx.qvel[:3]
    ball_vel = dx.qvel[3:5]
    return jnp.concatenate([joint_pos, pos_diff, ball_pos, joint_vel, ball_vel], axis=0)

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def control_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = dx.ctrl
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0.0001, 0.0001, 0.0001])), x))

def run_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    return jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([0, 0, 0, 4000, 4000, 0, 0, .1, .1, .1, 1, 1])), x)
    )

def terminal_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    return _cfg.dt * jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([0, 0, 0, 4000, 4000, 0, 0, .1, .1, .1, 1, 1])), x)
    )

def set_data(mx: mjx.Model, dx: mjx.Data, ctx: Context, key: jnp.ndarray) -> mjx.Data:
    ang1 = jax.random.uniform(key, (1,), minval=-0, maxval=2*jnp.pi)
    ang23 = jax.random.uniform(key, (2,), minval=-0.0, maxval=0.0)
    angs = jnp.concatenate([ang1, ang23], axis=0)
    l, d_theta = 0.5, 0.1
    rand_sign = jax.random.bernoulli(key, 0.5)
    obj_x = jax.random.uniform(key, (1,), minval=0.25, maxval=l) * (jnp.sin(angs[0] + d_theta * rand_sign))
    obj_y = jax.random.uniform(key, (1,), minval=0.25, maxval=l) * (jnp.cos(angs[0] + d_theta * rand_sign))
    obj_z = jax.random.uniform(key, (1,), minval=0.0, maxval=0.0)
    qpos = jnp.concatenate([angs, obj_x, obj_y, obj_z, jnp.array([1, 0, 0, 0])], axis=0)
    qvel = jax.random.uniform(key, (_cfg.mx.nv,), minval=-0.1, maxval=0.1)
    target_pos = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    target_pos = jnp.concatenate([target_pos, jnp.array([0])], axis=0)
    qpos = dx.qpos.at[:].set(qpos)
    qvel = dx.qvel.at[:].set(qvel)
    mocap_pos = dx.mocap_pos.at[:].set(target_pos)
    return dx.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap_pos)

def gen_network(seed: int) -> Network:
    key = jax.random.PRNGKey(seed)
    return Policy([12, 128, 128, _cfg.mx.nu], key)

def is_terminal(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    time_limit =  (dx.time/ mx.opt.timestep) > (_cfg.ntotal - 1)
    return jnp.array([time_limit])


ctx = Context(
    _cfg,
    Callbacks(
        run_cost=run_cost,
        terminal_cost=terminal_cost,
        control_cost=control_cost,
        set_data=set_data,
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_det,
        is_terminal=is_terminal
    )
)


