import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import equinox as eqx


class ValueFunction(eqx.Module):
    def __init__(self, dims: list, act):
        # use dims to define an eqx Sequential model with linear layers use act as activation
        _net = []
        for d in dims[:-2]:
            _net.append(eqx.nn.Linear(d, dims[dims.index(d) + 1]))
            _net.append(act)

        _net.append(eqx.nn.Linear(dims[-2], dims[-1]))
        self.net = eqx.nn.Sequential(*_net)

    def __call__(self, x):
        return self.net(x)


class Controller(object):
    def __init__(self, net):
        self._vf = net


    def __call__(self, x):
        return self._vf(x)


