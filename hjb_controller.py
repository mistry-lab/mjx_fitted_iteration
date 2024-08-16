import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx, mjtDataType
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
    def __init__(self, net, model, data):
        self._vf = net
        self._m = model
        self._dx = data

    # setter for the data
    @property
    def data(self):
        return self._dx

    @data.setter
    def data(self, data):
        self._dx = data

    def __call__(self, x):
        mass_mat = mjx.full_m(self._m, self._dx)
        # invert the mass matrix
        inv_mass_mat = jnp.linalg.inv(mass_mat)
        # compute derivative of the value function with respect to the state
        dvdx = jax.grad(self._vf)(x)
        u = -jnp.dot(inv_mass_mat, self._vf(x))

        return self._vf(x)


