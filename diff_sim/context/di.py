import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network

model_path = os.path.join(os.path.dirname(__file__), '../xmls/doubleintegrator.xml')

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
        return self.layers[-1](x)


def policy(
        x: jnp.ndarray, t: jnp.ndarray, net: Network, cfg: Config, mx: mjx.Model, dx: mjx.Data
) -> jnp.ndarray:
    # under this policy the cost function is quite important the cost that works is:
    # Q = diag([0, 0]) or Q = diag([10, 0.01]) and R = diag([0.01]) and QF = diag([10, 0.01])
    # act_id = mx.actuator_trnid[:, 0]
    # M = mjx.full_m(mx, dx)
    # invM = jnp.linalg.inv(M)
    # dvdx = jax.jacrev(net,0)(x, t)
    # G = jnp.vstack([jnp.zeros_like(invM), invM])
    # invR = jnp.linalg.inv(jnp.diag(jnp.array([0.01])))
    # u = (-1/2 * invR @ G.T[act_id, :] @ dvdx.T).flatten()
    return net(x, t)

def run_cost(x: jnp.ndarray) -> jnp.ndarray:
    # x^T Q x
    return  jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0, 0])), x))

def terminal_cost(x: jnp.ndarray) -> jnp.ndarray:
    # x^T Q_f x
    return 10*jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([10, 0.01])), x))

def control_cost(x: jnp.ndarray) -> jnp.ndarray:
    # u^T R u
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0.01])), x))

def init_gen(total_batch: int, key: jnp.ndarray) -> jnp.ndarray:
    xinits = jnp.concatenate([
        jax.random.uniform(key, (total_batch, 1), minval=-1, maxval=1),
        jax.random.uniform(key, (total_batch, 1), minval=-.7, maxval=.7)
    ], axis=1).squeeze()

    return xinits

def state_encoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def gen_network(seed: int) -> Network:
    key = jax.random.PRNGKey(seed)
    return Policy([3, 64, 64, 1], key)


ctx = Context(
    Config(
        lr=4e-3,
        num_gpu=1,
        seed=0,
        nsteps=100,
        epochs=1000,
        batch=64,
        samples=1,
        eval=10,
        dt=0.01,
        path=model_path,
        mx=mjx.put_model(mujoco.MjModel.from_xml_path(model_path)),
    ),
    Callbacks(
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