import jax
from jax import numpy as jnp
import equinox as eqx
from .config import Config, Callbacks, Context
import os
import jax.debug

try:
    # This works when __file__ is defined (e.g., in regular scripts)
    base_path = os.path.dirname(__file__)
except NameError:
    # Fallback to current working directory (e.g., in interactive sessions)
    base_path = os.getcwd()

base_path = os.path.join(base_path, '../xmls')

class ValueFunc(eqx.Module):
    layers: list
    act: lambda x: jax.nn.relu(x)

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i], use_bias=True) for i in range(len(dims) - 1)]
        self.act = jax.nn.softplus

        # new_weight = jnp.array([[1.5,1.], [1.,1.5]])
        # new_biases = jnp.array([0.,0.])
        # where = lambda l: l.weight
        # self.layers[0] = eqx.tree_at(where, self.layers[0], new_weight)

        # where = lambda l: l.bias
        # self.layers[0] = eqx.tree_at(where, self.layers[0], new_biases)

        # new_weight = jnp.array([1.,1.5])
        # new_biases = jnp.array([0.])
        # where = lambda l: l.weight
        # self.layers[1] = eqx.tree_at(where, self.layers[1], new_weight)
        # where = lambda l: l.bias
        # self.layers[1] = eqx.tree_at(where, self.layers[1], new_biases)
 
        # params = eqx.get_params(self.layers[0])
        # w1 = self.layers[0].weight.at[0].set(jnp.array([1.5,1.]))
        # self.layers[0] = self.layers[0].replace(weight=w1)
        # key1, subkey = jax.random.split(key)
        # self.layers2 = [eqx.nn.Linear(in_features=2, out_features=2, key=key1)]
        

    def __call__(self, x, t):
        x = jnp.concatenate([x, t], axis=-1)
        # transformed_x = self.layers2[0](x)
        # jax.debug.print(transformed_x)
        # return transformed_x @ x
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
            # x = layer(x)
        return self.layers[-1](x)


        # f = lambda x: jnp.einsum('...i,ij,...j->...', x, self.layers[0].weight, x)
        # v = f(x)
        # return v
    
    # PD controller
    # def __call__(self, x):
    #     f[ = lambda x: jnp.einsum('...i,ij,...j->...', x, jnp.array([[1.3, 1],1,1.3]]), x)
    #     v = f(x)
    #     return v

# class ValueFunc(eqx.Module):
#     layers: list

#     def __init__(self, key):
#         # Initialize a Linear layer with input size 2 and output size 2 (for 2x2 matrix)
#         key1, subkey = jax.random.split(key)
#         self.layers = [eqx.nn.Linear(in_features=2, out_features=2, key=key1)]  # Linear layer to parametrize Q
#         # self.linear = eqx.nn.Linear(in_features=2, out_features=2, key=key1)

#     def __call__(self, x):
#         # Apply the linear layer to transform the input x
#         # transformed_x = self.linear(x)
#         transformed_x = self.layers[0](x)
        
#         # Compute the quadratic form: (linear(x))^T * x
#         return jnp.dot(transformed_x, x)


ctx = Context(cfg=Config(
    model_path=os.path.join(base_path, 'doubleintegrator.xml'),
    dims=[4, 64, 64, 1],
    lr=1e-3,
    seed=0,
    nsteps=100,
    epochs=100,
    batch=100,
    vis=10,
    dt=0.01,
    R=jnp.array([[0.001]]),
    horizon=jnp.arange(0, 100 + 0.01, 0.01) + 0.01
    ),cbs=Callbacks(
        run_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([25, 0.25])), x),
        terminal_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([25, 0.25])), x),
        control_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.array([[0.001]]), x).at[..., -1].set(0),
        init_gen= lambda batch, key: jnp.concatenate([
            jax.random.uniform(key, (batch, 1), minval=-1., maxval=1.),
            jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1)
        ], axis=1).squeeze(),
    state_encoder=lambda x: x,
    net=ValueFunc([3, 64, 64, 1], jax.random.PRNGKey(0))
    # net=ValueFunc([2, 2, 1], jax.random.PRNGKey(0))
    )
)

