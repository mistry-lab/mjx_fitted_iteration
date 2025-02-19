import os
import jax
from PIL.ImageQt import qt_version
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import jax.numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
import optax

import diff_sim
from diff_sim.loss_funcs import loss_fn_policy_det
from diff_sim.simulation.simulate import make_simulate_fn_fd, make_simulate_fn
from diff_sim.context.meta_context import Context
from diff_sim.runner_fn import run


if __name__ == "__main__":
    # Load mj and mjx model
    model_path = os.path.join(os.path.dirname(diff_sim.__file__), "xmls", "cartpole.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    mx = mjx.put_model(model)

    class Policy(eqx.Module):
        layers: list
        act: callable
        dropout: callable

        def __init__(self, dims: list, key):
            keys = jax.random.split(key, len(dims))
            self.layers = [eqx.nn.Linear(
                dims[i], dims[i + 1], key=keys[i], use_bias=True
            ) for i in range(len(dims) - 1)]
            self.act = jax.nn.relu
            self.dropout = eqx.nn.Dropout(0.1)

        def __call__(self, x, key):
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self.act(x)
            x = self.layers[-1](x).squeeze()
            # x = jnp.tanh(x) * 1.
            return x


    def set_data(mx: mjx.Model, dx: mjx.Data, key: jnp.ndarray) -> mjx.Data:
        qpos = jnp.concatenate([
            jax.random.uniform(key, (1,), minval=-0.3, maxval=0.3),
            jax.random.uniform(key, (1,), minval=jnp.pi + 0.3, maxval=jnp.pi - 0.3),
        ]).squeeze()

        qvel = jnp.concatenate([
            jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1),
            jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1)
        ]).squeeze()

        dx = dx.replace(qpos=dx.qpos.at[:].set(qpos), qvel=dx.qvel.at[:].set(qvel))
        return dx

    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        return dx

    def gen_network(n: int) -> eqx.Module:
        key = jax.random.PRNGKey(n)
        return Policy([4, 64, 64, 1], key)

    def policy(net: eqx.Module, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
               ) -> tuple[mjx.Data, jnp.ndarray]:
        x = jnp.concatenate([dx.qpos, dx.qvel])
        u = net(x, policy_key)
        return dx, jnp.expand_dims(u, axis=0)

    def running_cost(mx: mjx.Model, dx: mjx.Data):
        u = dx.ctrl
        return 0.001 * jnp.sum(u ** 2)

    def terminal_cost(mx: mjx.Model, dx: mjx.Data):
        x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        return 10 * jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([200, 100, 2, 1])), x)).squeeze()

    # TODO: Add terminal time to is_terminal function
    # Is terminal function only for state, not time condition,
    # handled internally in time.
    def is_terminal(mx: mjx.Model, dx: mjx.Data):
        return jnp.array([False])


    ctx = Context(
        lr=3e-3,
        num_gpu=1,
        seed=0,
        nsteps=250,
        ntotal=250,
        epochs=200,
        batch=50,
        samples=1,
        eval=10,
        ctrl_dim=1,
        mx=mjx.put_model(model),
        gen_model=lambda: mujoco.MjModel.from_xml_path(model_path),
        gen_network=gen_network,
        run_cost=running_cost,
        terminal_cost=terminal_cost,
        set_data=set_data,
        set_control=set_control,
        controller=policy,
        is_terminal=is_terminal,
    )

    optimiser = optax.adamw(ctx.lr)
    simulate_fn = eqx.filter_jit(make_simulate_fn_fd(ctx))
    run(ctx, optimiser, simulate_fn, loss_fn_policy_det)
