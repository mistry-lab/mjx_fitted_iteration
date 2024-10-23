### this is the context file for the single arm robot

import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from jax.lib.xla_extension import pytree
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network
from diff_sim.utils.generic import save_model

model_path = os.path.join(os.path.dirname(__file__), '../xmls/single_arm.xml')

__cfg = Config(
    lr=1e-4,
    seed=0,
    batch=10,
    samples=1,
    epochs=1000,
    eval=10,
    num_gpu=1,
    path=model_path,
    dt=0.01,
    nsteps=350,
    mx=mjx.put_model(mujoco.MjModel.from_xml_path(model_path)),
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
        t = t if t.ndim == 1 else t.reshape(1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x).squeeze()
        # bound the control to be between -1 and 1 using tanh
        # x = jnp.tanh(x) * 2
        return x


def policy(
        x: jnp.ndarray, t: jnp.ndarray, net: Network, cfg: Config, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
    t = jnp.expand_dims(dx.time, axis=0)
    # noise = 0.1 * jax.random.normal(policy_key, (mx.nu,))
    u = net(x, t) # + noise
    dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
    return dx, u

def state_encoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def control_cost(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(x.T, x) * 0.01

def run_cost(x: jnp.ndarray) -> jnp.ndarray:
    # x^T Q x
    return  jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0, 0])), x))

def terminal_cost(x: jnp.ndarray) -> jnp.ndarray:
    # x^T Q_f x
    return 10*jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([1, 0.01])), x))

def init_gen(total_batch: int, key: jnp.ndarray) -> jnp.ndarray:
    return jax.random.uniform(key, (total_batch, __cfg.mx.nq + __cfg.mx.nv), minval=0, maxval=3.14)

def gen_network(seed: int) -> Network:
    key = jax.random.PRNGKey(seed)
    return Policy([__cfg.mx.nq + __cfg.mx.nv + 1, 64, 64, __cfg.mx.nu], key)


ctx = Context(
    cfg=__cfg,
    cbs=Callbacks(
        run_cost=run_cost,
        terminal_cost=terminal_cost,
        control_cost=control_cost,
        init_gen=init_gen,
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_det
    )
)


