import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox as eqx

class ValueFunc(eqx.Module):
    layers: list
    act: lambda x: jax.nn.relu(x)

    def __init__(self, dims: list, key, act):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(len(dims) - 1)]
        self.act = act

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class Controller(object):
    def __init__(self, dims, act, key):
        self.vf = ValueFunc(dims, key, act)

    def __call__(self, mx, dx):
        x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        act_id = mx.actuator_trnid[:, 0]
        M = mjx.full_m(mx, dx)
        invM = jnp.linalg.inv(M)
        dvdx = jax.jacrev(self.vf)(x)
        G = jnp.vstack([jnp.zeros_like(invM), invM])
        u = -(G.T @ dvdx.T)[act_id , :].squeeze()
        return u
